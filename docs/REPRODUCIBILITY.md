# Reproduction protocol

## Execution graph

```text
discovery
├── evaluation37
│   ├── ambiguity
│   ├── geometric_distance
│   └── matched_ablation
│       └── tie_sensitivity
│           ├── matched_familiarity
│           ├── matched_diagnostics ── ambiguity
│           └── matched_robustness ──── expanded46
└── expanded46
```

The runner constructs this graph from `Run` records in
`scripts/reproduce.py`. Every run has one configuration under `configs/` and
uses the shared implementation under `src/destination_prediction/`.

## Reused computations

Data preparation, catalogue construction, model fitting, and evaluation are
implemented once and parameterized by cohort configuration. The expanded
cohort omits familiarity calculations because no reported claim uses them.

Tie sensitivity produces the deterministic point-count full-grid and
last-state cases in one pass. Matched robustness reuses those cases and
computes only elapsed-time, cumulative-distance, and expanded-cohort cases.
The initial matched-ablation pass is retained because it supplies the exact
10,000-replicate bootstrap plan and the legacy-order reproduction audit.

## Deterministic decisions

- Cohorts, ratios, configurations, seeds, and bootstrap parameters are frozen
  in JSON.
- Equal-distance candidates are ordered by immutable training trajectory ID.
- Equal destination mass is resolved by training-catalogue order.
- Bootstrap draws preserve complete four-ratio case vectors.
- The last-state ablation changes only whether preceding aligned states enter
  the grid distance.

## Run outputs

Each run under `work/runs/<run-id>/` contains a portable `run_manifest.json`,
per-stage state markers, logs, generated outputs, validation report, and a
`COMPLETE.json` marker written only after all validations pass. Cached stages
are reused only when their configuration and shared-source fingerprint match.

The immutable reference execution is stored under `reference/`: dataset,
normalized cohort, run-role, and environment records in `manifests/`,
coordinate-free cases in `predictions/`, and validated statistics in
`results/`.
