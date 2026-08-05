#!/usr/bin/env python3
"""Run the public TSMini encoder with its kNN-guided ranking objective."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from destination_prediction.catalogue import build_catalogue
from destination_prediction.context import RunContext, implementation_constants
from destination_prediction.data import add_local_sequences, load_marked, partitions, user_origins
from destination_prediction.geometry import prefix_cutoff, resample_path
from destination_prediction.metrics import evaluate_native_predictions, selection_summary


TSMINI_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "TSMini"
sys.path.insert(0, str(TSMINI_ROOT))

from config import Config  # noqa: E402
from model.lambdaloss import lambdaLoss  # noqa: E402
from model.tsmini import TSMini  # noqa: E402
from utils.traj import padding_traj, preprocess_traj  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure(embedding_dim: int, normalizer_m: float, config: dict) -> None:
    architecture = implementation_constants(
        config, "tsmini_inspired_retrieval"
    )["public_architecture"]
    Config.device = torch.device("cpu")
    Config.min_traj_len = 7
    Config.max_traj_len = int(config["dataset"]["max_points_after_even_downsampling"])
    Config.traj_distance_norm_denominator = float(normalizer_m)
    Config.traj_duplicate_short_tolerance = 0
    Config.seq_embedding_dim = int(embedding_dim)
    Config.tsmini_conv_channel_dim = int(architecture["convolution_channels"])
    Config.tsmini_conv_hidden_in_dim = int(architecture["convolution_hidden_dim"])
    Config.tsmini_conv_kernel_size = int(architecture["convolution_kernel_size"])
    Config.tsmini_conv_stride = int(architecture["convolution_stride"])
    Config.tsmini_patch_emb_dim = int(embedding_dim)
    Config.tsmin_trans_attention_head = int(architecture["attention_heads"])
    Config.tsmin_trans_attention_layer = int(architecture["attention_layers"])
    Config.tsmin_trans_hidden_dim = int(architecture["transformer_hidden_dim"])
    Config.tsmin_trans_attention_dropout = float(architecture["attention_dropout"])
    Config.tsmin_trans_pos_encoder_dropout = float(architecture["attention_dropout"])


def prefix_sequences(frame: pd.DataFrame, ratio: float) -> list[list[list[float]]]:
    return [seq[: prefix_cutoff(len(seq), ratio)] for seq in frame["local_seq"]]


def space_from_train(train: pd.DataFrame, config: dict) -> dict[str, float]:
    points = np.asarray([point for seq in train["local_seq"] for point in seq], dtype=float)
    buffer_m = float(
        implementation_constants(config, "tsmini_inspired_retrieval")[
            "cellspace_buffer_m"
        ]
    )
    return {
        "x_min": float(points[:, 0].min() - buffer_m),
        "x_max": float(points[:, 0].max() + buffer_m),
        "y_min": float(points[:, 1].min() - buffer_m),
        "y_max": float(points[:, 1].max() + buffer_m),
    }


def collate(sequences: list[list[list[float]]], space: dict[str, float]):
    processed = [preprocess_traj(seq, space, 0) for seq in sequences]
    processed = [features + [features[-1]] * max(0, 7 - len(features)) for features in processed]
    padded, lengths = padding_traj(processed)
    features = torch.tensor(padded, dtype=torch.float32)
    features[:, :, 2:4] = features[:, :, 2:4].clamp(0.0, 1.0)
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite TSMini input")
    return features, torch.tensor(lengths, dtype=torch.long)


def target_distance(
    sequences: list[list[list[float]]], normalizer_m: float, resample_points: int
) -> torch.Tensor:
    arrays = np.stack([resample_path(seq, resample_points) for seq in sequences])
    distances = np.linalg.norm(arrays[:, None, :, :] - arrays[None, :, :, :], axis=3).mean(axis=2)
    distances = np.clip(distances / float(normalizer_m), 0.0, 1.0)
    return torch.tensor(distances, dtype=torch.float32)


def official_loss(embeddings: torch.Tensor, truth: torch.Tensor, ranking_k: int, mse_weight: float):
    mask = torch.triu(torch.ones_like(truth, dtype=torch.bool), diagonal=1)
    truth_vector = truth[mask]
    prediction_matrix = torch.cdist(embeddings, embeddings, p=1)
    prediction_vector = prediction_matrix[mask]
    scale = truth_vector.mean() / prediction_vector.mean().clamp_min(1e-8)
    prediction_vector = prediction_vector * scale
    loss_wmse = torch.mean(torch.square(prediction_vector - truth_vector) * (1.0 - truth_vector))

    n = prediction_matrix.shape[0]
    prediction_max = prediction_matrix.max(dim=-1)[0].unsqueeze(1)
    predicted_rank = prediction_matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
    predicted_rank = prediction_max - predicted_rank
    truth_max = truth.max(dim=-1)[0].unsqueeze(1)
    truth_rank = truth.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
    truth_rank = truth_max - truth_rank
    loss_rank = lambdaLoss(predicted_rank, truth_rank, k=min(int(ranking_k), n - 1))
    total = loss_wmse * float(mse_weight) * 100.0 + loss_rank * (1.0 - float(mse_weight))
    return total, loss_wmse, loss_rank


def training_pool(train: pd.DataFrame, ratios: list[float]):
    sequences = []
    for ratio in ratios:
        sequences.extend(prefix_sequences(train, ratio))
    return sequences


def train_one_epoch(model, sequences, space, optimizer, cfg, rng):
    model.train()
    batch_size = int(cfg["batch_size"])
    order = rng.permutation(len(sequences))
    losses = []
    for start in range(0, len(order) - batch_size + 1, batch_size):
        indices = order[start : start + batch_size]
        batch = [sequences[int(index)] for index in indices]
        features, lengths = collate(batch, space)
        truth = target_distance(
            batch,
            float(Config.traj_distance_norm_denominator),
            int(cfg["target_resample_points"]),
        )
        optimizer.zero_grad()
        embeddings = model(features, lengths)
        loss, wmse, ranking = official_loss(
            embeddings,
            truth,
            int(cfg["ranking_k"]),
            float(cfg["weighted_mse_loss_weight"]),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite TSMini loss")
        loss.backward()
        optimizer.step()
        losses.append((float(loss.item()), float(wmse.item()), float(ranking.item())))
    return np.mean(np.asarray(losses), axis=0).tolist()


@torch.no_grad()
def embed(model, sequences, space, batch_size: int = 128) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(sequences), batch_size):
        features, lengths = collate(sequences[start : start + batch_size], space)
        output.append(model(features, lengths).cpu().numpy())
    return np.vstack(output)


def retrieval_predictions(
    model,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    ratios: list[float],
    space: dict[str, float],
    config_id: str,
    seed: int,
    target_resample_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fidelity_rows = []
    train_users = train["user_id"].to_numpy(dtype=int)
    train_indices = {
        int(user_id): np.flatnonzero(train_users == int(user_id))
        for user_id in sorted(np.unique(train_users))
    }
    for ratio in ratios:
        train_sequences = prefix_sequences(train, ratio)
        eval_sequences = prefix_sequences(evaluate, ratio)
        train_embeddings = embed(model, train_sequences, space)
        eval_embeddings = embed(model, eval_sequences, space)
        for eval_index, row in enumerate(evaluate.itertuples(index=False)):
            candidates = train_indices[int(row.user_id)]
            start = time.perf_counter()
            learned = np.abs(train_embeddings[candidates] - eval_embeddings[eval_index]).sum(axis=1)
            choice = int(candidates[int(np.argmin(learned))])
            runtime_ms = (time.perf_counter() - start) * 1000.0
            analogue = train.iloc[choice]
            lon, lat = analogue["wgs_seq"][-1]
            rows.append(
                {
                    "family": "tsmini_retrieval",
                    "config_id": config_id,
                    "seed": int(seed),
                    "ratio": float(ratio),
                    "user_id": int(row.user_id),
                    "trip_id": str(row.trip_id),
                    "native_pred_lon": float(lon),
                    "native_pred_lat": float(lat),
                    "analogue_trip_id": str(analogue["trip_id"]),
                    "query_runtime_ms": runtime_ms,
                }
            )
            points = int(target_resample_points)
            query_geom = resample_path(eval_sequences[eval_index], points)
            candidate_geom = np.stack(
                [resample_path(train_sequences[int(idx)], points) for idx in candidates]
            )
            exact = np.linalg.norm(candidate_geom - query_geom[None, :, :], axis=2).mean(axis=1)
            k = min(10, len(candidates))
            learned_top = set(np.argsort(learned)[:k].tolist())
            exact_top = set(np.argsort(exact)[:k].tolist())
            fidelity_rows.append(
                {
                    "config_id": config_id,
                    "seed": int(seed),
                    "ratio": float(ratio),
                    "user_id": int(row.user_id),
                    "trip_id": str(row.trip_id),
                    "hr_at_10": len(learned_top.intersection(exact_top)) / k,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fidelity_rows)


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
    return train, evaluate, catalog, space_from_train(train, config)


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
            **fixed["tsmini"],
            "selection_source": "frozen 30-user discovery validation",
            "outer_test_read": False,
        }

    selection_seed = int(config["selection"]["selection_seed"])
    train, validation, fit_catalog, space = prepare_stage(marked, "inner", config)
    fit_catalog.to_csv(output / "fit_only_destination_catalog.csv", index=False)
    pool = training_pool(train, ratios)
    budgets = sorted(map(int, cfg["epoch_budgets"]))
    validation_frames = []
    fidelity_frames = []
    histories = []
    for embedding_dim in map(int, cfg["sequence_embedding_dim"]):
        for normalizer_m in map(float, cfg["trajectory_distance_normalizer_m"]):
            set_seed(selection_seed)
            configure(embedding_dim, normalizer_m, config)
            model = TSMini()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(cfg["learning_rate"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            rng = np.random.default_rng(selection_seed)
            for epoch in range(1, max(budgets) + 1):
                total, wmse, ranking = train_one_epoch(
                    model, pool, space, optimizer, cfg, rng
                )
                histories.append(
                    {
                        "embedding_dim": embedding_dim,
                        "normalizer_m": normalizer_m,
                        "epoch": epoch,
                        "loss": total,
                        "weighted_mse": wmse,
                        "ranking_loss": ranking,
                    }
                )
                if epoch in budgets:
                    config_id = f"e{embedding_dim}_d{int(normalizer_m)}_epochs{epoch}"
                    predictions, fidelity = retrieval_predictions(
                        model,
                        train,
                        validation,
                        ratios,
                        space,
                        config_id,
                        selection_seed,
                        int(cfg["target_resample_points"]),
                    )
                    validation_frames.append(
                        predictions.assign(
                            embedding_dim=embedding_dim,
                            normalizer_m=normalizer_m,
                            epoch_budget=epoch,
                        )
                    )
                    fidelity_frames.append(fidelity)
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
        ["config_id", "embedding_dim", "normalizer_m", "epoch_budget"],
    )
    validation_evaluated.to_csv(output / "validation_predictions.csv", index=False)
    per_user.to_csv(output / "validation_by_user_ratio.csv", index=False)
    leaderboard.to_csv(output / "validation_leaderboard.csv", index=False)
    pd.concat(fidelity_frames, ignore_index=True).to_csv(
        output / "validation_neighbor_fidelity.csv", index=False
    )
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
    output = context.run_directory / "outputs" / "tsmini"
    output.mkdir(parents=True, exist_ok=True)
    marked = load_marked(context.run_directory)
    ratios = list(map(float, config["task"]["observation_ratios"]))
    cfg = implementation_constants(config, "tsmini_inspired_retrieval")
    selected = selected_configuration(output, marked, ratios, cfg, config)
    (output / "selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )

    train, test, catalog, space = prepare_stage(marked, "outer", config)
    catalog.to_csv(output / "outer_training_destination_catalog.csv", index=False)
    pool = training_pool(train, ratios)
    final_frames = []
    final_fidelity = []
    final_history = []
    model_sizes = []
    for seed in map(int, config["selection"]["final_neural_seeds"]):
        set_seed(seed)
        configure(
            int(selected["embedding_dim"]), float(selected["normalizer_m"]), config
        )
        model = TSMini()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        rng = np.random.default_rng(seed)
        for epoch in range(1, int(selected["epoch_budget"]) + 1):
            total, wmse, ranking = train_one_epoch(model, pool, space, optimizer, cfg, rng)
            final_history.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "loss": total,
                    "weighted_mse": wmse,
                    "ranking_loss": ranking,
                }
            )
        predictions, fidelity = retrieval_predictions(
            model,
            train,
            test,
            ratios,
            space,
            str(selected["config_id"]),
            seed,
            int(cfg["target_resample_points"]),
        )
        final_frames.append(predictions)
        final_fidelity.append(fidelity)
        model_sizes.append(
            {
                "seed": seed,
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "parameter_bytes": int(
                    sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
                ),
            }
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
    pd.concat(final_fidelity, ignore_index=True).to_csv(
        output / "test_neighbor_fidelity_by_seed.csv", index=False
    )
    pd.DataFrame(final_history).to_csv(output / "final_training_history.csv", index=False)
    pd.DataFrame(model_sizes).to_csv(output / "model_size_by_seed.csv", index=False)
    summary = {
        "implementation": "public TSMini architecture and kNN-guided loss",
        "similarity_target": "normalized 32-point arc-length-resampled prefix L2 distance",
        "selected": selected,
        "final_seeds": config["selection"]["final_neural_seeds"],
        "test_labels_used_for_selection": False,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = run(RunContext.from_environment())
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
