#!/usr/bin/env python3
"""Verify the frozen scientific protocol and normalized cohort allocation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    discovery = json.loads((ROOT / "configs" / "discovery.json").read_text())
    evaluation = json.loads((ROOT / "configs" / "evaluation37.json").read_text())
    expanded = json.loads((ROOT / "configs" / "expanded46.json").read_text())
    with (ROOT / "reference" / "manifests" / "cohort_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        cohorts = list(csv.DictReader(handle))

    discovery_users = {
        int(row["user_id"]) for row in cohorts if row["discovery_membership"] == "true"
    }
    evaluation_users = {
        int(row["user_id"]) for row in cohorts if row["evaluation_membership"] == "true"
    }
    expanded_users = {
        int(row["user_id"]) for row in cohorts if row["expanded_membership"] == "true"
    }
    eligible_users = {
        int(row["user_id"]) for row in cohorts if row["eligibility"] == "true"
    }
    eligible_prior_users = {
        int(row["user_id"])
        for row in cohorts
        if row["eligibility"] == "true" and row["prior_use"] == "true"
    }
    assert len(cohorts) == 182
    assert len({int(row["user_id"]) for row in cohorts}) == 182
    assert len(eligible_users) == 76
    assert len(discovery_users) == discovery["dataset"]["cohort_users"] == 30
    assert len(evaluation_users) == evaluation["dataset"]["cohort_users"] == 37
    assert len(expanded_users) == expanded["dataset"]["cohort_users"] == 46
    assert discovery_users.isdisjoint(evaluation_users)
    assert discovery_users | evaluation_users | (expanded_users - evaluation_users) == eligible_users
    assert len(eligible_prior_users - discovery_users) == 9
    assert expanded_users - evaluation_users == eligible_prior_users - discovery_users
    assert evaluation_users == set(evaluation["dataset"]["include_user_ids"])
    assert expanded_users == evaluation_users | (expanded_users - evaluation_users)
    assert len(expanded_users - evaluation_users) == 9
    assert all(
        row["prior_use"] == "false"
        for row in cohorts
        if row["evaluation_membership"] == "true"
    )

    assert evaluation["task"]["observation_ratios"] == [0.25, 0.5, 0.66, 0.75]
    assert evaluation["task"]["primary_dbscan_eps_m"] == 200.0
    assert evaluation["selection"]["final_neural_seeds"] == [20260723, 20260724, 20260725, 20260726, 20260727]
    bootstrap = evaluation["evaluation"]["bootstrap"]
    assert bootstrap["design"] == "paired hierarchical user-then-trip bootstrap"
    assert bootstrap["replicates"] == 10000
    assert bootstrap["seed"] == 20260723
    assert bootstrap["confidence_level"] == 0.95
    print(
        json.dumps(
            {
                "status": "passed",
                "source_users": len(cohorts),
                "eligible_users": len(eligible_users),
                "discovery_users": len(discovery_users),
                "frozen_evaluation_users": len(evaluation_users),
                "overlapping_expanded_users": len(expanded_users),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
