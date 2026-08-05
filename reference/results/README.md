# Frozen reporting results

This directory contains validated, publication-facing outputs from the
reference runs. It excludes coordinate catalogues, preprocessed trajectories,
model checkpoints, runtime logs, and duplicated case-level files.

The initial matched-ablation directory is retained because it holds the exact
10,000-replicate hierarchical bootstrap plan reused by later analyses. Its
legacy equal-distance result is superseded by `tie_sensitivity/id_top10` and
`matched_robustness`; this provenance is stated in `docs/CLAIM_MAP.md` and
`../manifests/run_registry.json`.

Case-level outcomes live in `../predictions/`. Human-readable extracts are
generated under `../../results/paper_tables/` by
`../../scripts/export_paper_tables.py`.
