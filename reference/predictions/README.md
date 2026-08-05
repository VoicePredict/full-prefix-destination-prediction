# Coordinate-free prediction files

All files are gzip-compressed CSV. They retain the case-level outcomes needed
for metric aggregation and paired comparisons, but contain no longitude,
latitude, raw sequence, or runtime column. `case_id` and `analogue_case_id`
are stable SHA-256-derived identifiers used consistently across released
tables; they are not source filenames.

## Files

- `cross_method_cases.csv.gz`: all representative methods, baselines, seeds,
  catalogue sensitivities, coverage, Hit@R80/R90/R95, D200, and D1km.
- `matched_grid_cases.csv.gz`: initial matched full-grid/last-state case matrix
  retained for the saved bootstrap order.
- `tie_sensitivity_cases.csv.gz`: deterministic, inclusive-boundary, and
  top-5/top-10/top-20 matched-grid variants.
- `tie_boundary_cases.csv.gz`: number and prevalence of candidates tied at
  the truncation boundary.
- `matched_familiarity_cases.csv.gz`: matched outcomes joined to the
  training-only close-analogue indicator.
- `matched_robustness_evaluation37_cases.csv.gz`: point-count, elapsed-time,
  and cumulative-distance matched predictions for the evaluation cohort.
- `matched_robustness_expanded46_cases.csv.gz`: point-count matched predictions
  for the overlapping expanded sensitivity cohort.
- `geometric_distance_cases.csv.gz`: aligned pointwise, DTW, and discrete
  Fréchet retrieval outcomes.
- `ambiguity_binned_cases.csv.gz`: training-only location-ambiguity features.
- `ambiguity_prediction_cases.csv.gz`: prediction outcomes joined to those
  ambiguity features.

Boolean metric columns use `0` and `1`. `covered_r90=0` denotes an endpoint
outside every training-derived R90 destination support; it remains in the
denominator and `hit_r90_all=0`. Distance errors are metres. Observation
ratios are fractions (`0.25`, `0.50`, `0.66`, and `0.75`).

To load a file with pandas:

```python
import pandas as pd

cases = pd.read_csv("reference/predictions/tie_sensitivity_cases.csv.gz")
```
