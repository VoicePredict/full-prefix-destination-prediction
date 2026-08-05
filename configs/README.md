# Frozen configurations

One JSON document defines each scientific run. Configurations contain cohort
membership, chronological splitting, observation ratios, method parameters,
seeds, bootstrap design, analysis status, and parameter rationales.

Dependencies use the short run IDs `discovery`, `evaluation37`, `expanded46`,
`ambiguity`, `geometric_distance`, `matched_ablation`, `tie_sensitivity`,
`matched_familiarity`, `matched_diagnostics`, and `matched_robustness`. The
role and status of every run are listed in
`reference/manifests/run_registry.json`; no configuration refers to a private
machine or a historical experiment directory.
