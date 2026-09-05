#!/usr/bin/env python3
"""Tune one common-catalog GRU structure and refit it under five fixed seeds."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from destination_prediction.catalogue import build_catalogue, endpoint_catalogue_assignment
from destination_prediction.context import RunContext, implementation_constants, parse_run_context
from destination_prediction.data import add_local_sequences, load_marked, partitions, user_origins
from destination_prediction.geometry import prefix_cutoff
from destination_prediction.metrics import evaluate_native_predictions, selection_summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def user_scales(train: pd.DataFrame) -> dict[int, np.ndarray]:
    scales: dict[int, np.ndarray] = {}
    for user_id, group in train.groupby("user_id", sort=True):
        points = np.asarray([point for seq in group["local_seq"] for point in seq], dtype=float)
        scales[int(user_id)] = np.maximum(points.std(axis=0), 1.0)
    return scales


def feature_sequence(seq: list[list[float]], scale: np.ndarray, ratio: float) -> np.ndarray:
    cutoff = prefix_cutoff(len(seq), ratio)
    xy = np.asarray(seq[:cutoff], dtype=np.float32) / scale.astype(np.float32)
    delta = np.vstack([np.zeros((1, 2), dtype=np.float32), np.diff(xy, axis=0)])
    progress = np.linspace(0.0, 1.0, cutoff, dtype=np.float32)[:, None]
    return np.hstack([xy, delta, progress]).astype(np.float32)


@dataclass(frozen=True)
class Case:
    user_id: int
    trip_id: str
    ratio: float
    features: np.ndarray
    target: int


class PrefixDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        ratios: list[float],
        scales: dict[int, np.ndarray],
        user_index: dict[int, int],
        label_map: dict[tuple[int, str], int] | None,
    ) -> None:
        self.cases: list[Case] = []
        for row in frame.itertuples(index=False):
            key = (int(row.user_id), str(row.trip_id))
            target = -1 if label_map is None else int(label_map[key])
            for ratio in ratios:
                self.cases.append(
                    Case(
                        user_id=int(row.user_id),
                        trip_id=str(row.trip_id),
                        ratio=float(ratio),
                        features=feature_sequence(row.local_seq, scales[int(row.user_id)], ratio),
                        target=target,
                    )
                )
        self.user_index = user_index

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        case = self.cases[index]
        return (
            torch.tensor(case.features, dtype=torch.float32),
            int(self.user_index[case.user_id]),
            int(case.target),
            case.user_id,
            case.trip_id,
            case.ratio,
        )


def collate(batch):
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    features = torch.zeros((len(batch), maximum, 5), dtype=torch.float32)
    for index, item in enumerate(batch):
        features[index, : len(item[0])] = item[0]
    return (
        features,
        lengths,
        torch.tensor([item[1] for item in batch], dtype=torch.long),
        torch.tensor([item[2] for item in batch], dtype=torch.long),
        [item[3] for item in batch],
        [item[4] for item in batch],
        [item[5] for item in batch],
    )


class DestinationGRU(nn.Module):
    def __init__(
        self,
        users: int,
        classes: int,
        hidden: int,
        user_embedding: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=5,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.user_embedding = nn.Embedding(users, user_embedding)
        output = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(output + user_embedding, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, features, lengths, user_index, class_user_index):
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.gru(packed)
        if self.gru.bidirectional:
            encoded = torch.cat([state[-2], state[-1]], dim=1)
        else:
            encoded = state[-1]
        logits = self.head(torch.cat([encoded, self.user_embedding(user_index)], dim=1))
        mask = class_user_index[None, :] != user_index[:, None]
        return logits.masked_fill(mask, -1e9)


def class_tables(train: pd.DataFrame, catalog: pd.DataFrame):
    keys = [
        (int(row.user_id), int(row.center_id))
        for row in catalog.sort_values(["user_id", "center_id"]).itertuples(index=False)
    ]
    class_index = {key: idx for idx, key in enumerate(keys)}
    cluster_map = endpoint_catalogue_assignment(train, catalog)
    labels = {
        trip_key: class_index[(trip_key[0], center_id)]
        for trip_key, center_id in cluster_map.items()
    }
    centers = {
        idx: (
            float(catalog_row.medoid_lon),
            float(catalog_row.medoid_lat),
        )
        for idx, catalog_row in enumerate(
            catalog.sort_values(["user_id", "center_id"]).itertuples(index=False)
        )
    }
    return keys, labels, centers


def train_epoch(model, loader, optimizer, class_user_index, gradient_norm_clip: float) -> float:
    model.train()
    losses = []
    for features, lengths, users, targets, *_ in loader:
        optimizer.zero_grad()
        logits = model(features, lengths, users, class_user_index)
        loss = nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite recurrent-classifier loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, dataset, class_user_index, center_by_class, family, config_id, seed, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    rows = []
    for features, lengths, users, _, user_ids, trip_ids, ratios in loader:
        logits = model(features, lengths, users, class_user_index)
        predicted = logits.argmax(dim=1).cpu().numpy()
        for position, class_id in enumerate(predicted):
            lon, lat = center_by_class[int(class_id)]
            rows.append(
                {
                    "family": family,
                    "config_id": config_id,
                    "seed": int(seed),
                    "ratio": float(ratios[position]),
                    "user_id": int(user_ids[position]),
                    "trip_id": str(trip_ids[position]),
                    "native_pred_lon": lon,
                    "native_pred_lat": lat,
                }
            )
    return pd.DataFrame(rows)


def prepare_stage(marked: pd.DataFrame, stage: str, config: dict):
    train, evaluate = partitions(marked, stage)
    origins = user_origins(train)
    train = add_local_sequences(train, origins)
    evaluate = add_local_sequences(evaluate, origins)
    catalog = build_catalogue(
        train,
        float(config["task"]["primary_dbscan_eps_m"]),
        list(map(float, config["evaluation"]["support_quantiles"])),
        int(config["task"]["dbscan_min_samples"]),
    )
    scales = user_scales(train)
    users = sorted(map(int, train["user_id"].unique()))
    user_index = {user_id: idx for idx, user_id in enumerate(users)}
    keys, labels, centers = class_tables(train, catalog)
    class_users = torch.tensor([user_index[user_id] for user_id, _ in keys], dtype=torch.long)
    return train, evaluate, catalog, scales, user_index, labels, centers, class_users


def selected_configuration(
    output: Path,
    marked: pd.DataFrame,
    ratios: list[float],
    cfg: dict[str, object],
    config: dict,
) -> dict[str, object]:
    fixed = config["selection"].get("fixed_configs")
    if fixed is not None:
        return {
            **fixed["recurrent"],
            "selection_source": "frozen 30-user discovery validation",
            "outer_test_read": False,
        }

    selection_seed = int(config["selection"]["selection_seed"])
    set_seed(selection_seed)
    train, validation, fit_catalog, scales, users, labels, centers, class_users = prepare_stage(
        marked, "inner", config
    )
    fit_catalog.to_csv(output / "fit_only_destination_catalog.csv", index=False)
    train_dataset = PrefixDataset(train, ratios, scales, users, labels)
    validation_dataset = PrefixDataset(validation, ratios, scales, users, None)
    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(selection_seed),
    )
    validation_frames = []
    histories = []
    budgets = sorted(map(int, cfg["epoch_budgets"]))
    for hidden in map(int, cfg["hidden_dim"]):
        for direction in cfg["direction"]:
            set_seed(selection_seed)
            model = DestinationGRU(
                users=len(users),
                classes=len(centers),
                hidden=hidden,
                user_embedding=int(cfg["user_embedding_dim"]),
                bidirectional=(direction == "bi"),
                dropout=float(cfg["dropout"]),
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(cfg["learning_rate"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            for epoch in range(1, max(budgets) + 1):
                loss = train_epoch(
                    model, loader, optimizer, class_users, float(cfg["gradient_norm_clip"])
                )
                histories.append(
                    {
                        "hidden_dim": hidden,
                        "direction": direction,
                        "epoch": epoch,
                        "training_loss": loss,
                    }
                )
                if epoch in budgets:
                    config_id = f"{direction}_h{hidden}_e{epoch}"
                    validation_frames.append(
                        predict(
                            model,
                            validation_dataset,
                            class_users,
                            centers,
                            "recurrent_destination_classifier",
                            config_id,
                            selection_seed,
                            int(cfg["batch_size"]),
                        ).assign(
                            hidden_dim=hidden,
                            direction=direction,
                            epoch_budget=epoch,
                        )
                    )
    validation_native = pd.concat(validation_frames, ignore_index=True)
    validation_evaluated = evaluate_native_predictions(
        validation_native,
        validation,
        fit_catalog,
        list(map(float, config["evaluation"]["support_quantiles"])),
        list(map(float, config["evaluation"]["fixed_radius_m"])),
        list(map(float, config["evaluation"]["point_thresholds_m"])),
    )
    per_user, leaderboard = selection_summary(
        validation_evaluated,
        ["config_id", "hidden_dim", "direction", "epoch_budget"],
    )
    validation_evaluated.to_csv(output / "validation_predictions.csv", index=False)
    per_user.to_csv(output / "validation_by_user_ratio.csv", index=False)
    leaderboard.to_csv(output / "validation_leaderboard.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
    selected = leaderboard.iloc[0].to_dict()
    selected.update(
        {
            "selection_seed": selection_seed,
            "selection_metric": config["selection"]["primary_metric"],
            "outer_test_read": False,
        }
    )
    return selected


def run(context: RunContext) -> dict[str, object]:
    config = context.config
    context.require_fixed_configurations_match()
    output = context.run_directory / "outputs" / "recurrent"
    output.mkdir(parents=True, exist_ok=True)
    marked = load_marked(context.run_directory)
    ratios = list(map(float, config["task"]["observation_ratios"]))
    cfg = implementation_constants(config, "recurrent_destination_classifier")
    selected = selected_configuration(output, marked, ratios, cfg, config)
    (output / "selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )

    train, test, catalog, scales, users, labels, centers, class_users = prepare_stage(
        marked, "outer", config
    )
    catalog.to_csv(output / "outer_training_destination_catalog.csv", index=False)
    train_dataset = PrefixDataset(train, ratios, scales, users, labels)
    test_dataset = PrefixDataset(test, ratios, scales, users, None)
    final_frames = []
    final_history = []
    for seed in map(int, config["selection"]["final_neural_seeds"]):
        set_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )
        model = DestinationGRU(
            users=len(users),
            classes=len(centers),
            hidden=int(selected["hidden_dim"]),
            user_embedding=int(cfg["user_embedding_dim"]),
            bidirectional=(str(selected["direction"]) == "bi"),
            dropout=float(cfg["dropout"]),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
        )
        for epoch in range(1, int(selected["epoch_budget"]) + 1):
            loss = train_epoch(
                model, loader, optimizer, class_users, float(cfg["gradient_norm_clip"])
            )
            final_history.append({"seed": seed, "epoch": epoch, "training_loss": loss})
        final_frames.append(
            predict(
                model,
                test_dataset,
                class_users,
                centers,
                "recurrent_destination_classifier",
                str(selected["config_id"]),
                seed,
                int(cfg["batch_size"]),
            )
        )
    final_native = pd.concat(final_frames, ignore_index=True)
    final_evaluated = evaluate_native_predictions(
        final_native,
        test,
        catalog,
        list(map(float, config["evaluation"]["support_quantiles"])),
        list(map(float, config["evaluation"]["fixed_radius_m"])),
        list(map(float, config["evaluation"]["point_thresholds_m"])),
    )
    final_evaluated.to_csv(output / "test_predictions_by_seed.csv", index=False)
    pd.DataFrame(final_history).to_csv(output / "final_training_history.csv", index=False)
    summary = {
        "fit_trips": int((marked["partition"] == "fit").sum()),
        "validation_trips": int((marked["partition"] == "validation").sum()),
        "outer_training_trips": int(marked["partition"].isin(["fit", "validation"]).sum()),
        "test_trips": int((marked["partition"] == "test").sum()),
        "selected": selected,
        "final_seeds": config["selection"]["final_neural_seeds"],
        "test_labels_used_for_selection": False,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = run(parse_run_context(description=__doc__))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
