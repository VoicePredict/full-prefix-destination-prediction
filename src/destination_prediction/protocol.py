"""Configuration-bound scientific protocol used by dependent analyses."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from destination_prediction.catalogue import (
    build_catalogue,
    endpoint_cluster_map,
    nearest_catalogue_center,
)
from destination_prediction.context import RunContext
from destination_prediction.data import (
    add_local_sequences,
    load_marked,
    partitions,
    user_origins,
)
from destination_prediction.geometry import (
    EARTH_RADIUS_M,
    haversine_m,
    local_xy,
    prefix_cutoff,
    resample_path,
)
from destination_prediction.metrics import evaluate_native_predictions, selection_summary


@dataclass(frozen=True)
class EvaluationProtocol:
    """Bind pure scientific operations to one run's frozen configuration."""

    context: RunContext

    @property
    def config(self) -> dict:
        return self.context.config

    @property
    def CONFIG(self) -> dict:  # retained for exact analysis equations
        return self.config

    def load_marked(self) -> pd.DataFrame:
        return load_marked(self.context.run_directory)

    @staticmethod
    def partitions(marked: pd.DataFrame, stage: str):
        return partitions(marked, stage)

    @staticmethod
    def user_origins(train: pd.DataFrame):
        return user_origins(train)

    @staticmethod
    def add_local_sequences(frame: pd.DataFrame, origins):
        return add_local_sequences(frame, origins)

    def build_catalogue(self, train: pd.DataFrame, eps_m: float) -> pd.DataFrame:
        return build_catalogue(
            train,
            eps_m,
            list(map(float, self.config["evaluation"]["support_quantiles"])),
            int(self.config["task"]["dbscan_min_samples"]),
        )

    build_catalog = build_catalogue

    @staticmethod
    def endpoint_cluster_map(train: pd.DataFrame, catalogue: pd.DataFrame):
        return endpoint_cluster_map(train, catalogue)

    def evaluate_native_predictions(
        self, native: pd.DataFrame, truth: pd.DataFrame, catalogue: pd.DataFrame
    ) -> pd.DataFrame:
        evaluation = self.config["evaluation"]
        return evaluate_native_predictions(
            native,
            truth,
            catalogue,
            list(map(float, evaluation["support_quantiles"])),
            list(map(float, evaluation["fixed_radius_m"])),
            list(map(float, evaluation["point_thresholds_m"])),
        )

    prefix_cutoff = staticmethod(prefix_cutoff)
    resample_path = staticmethod(resample_path)
    local_xy = staticmethod(local_xy)
    haversine_m = staticmethod(haversine_m)
    nearest_catalogue_center = staticmethod(nearest_catalogue_center)
    nearest_catalog_center = staticmethod(nearest_catalogue_center)
    selection_summary = staticmethod(selection_summary)


__all__ = ["EARTH_RADIUS_M", "EvaluationProtocol"]
