# Coordinate-free prediction files

All files are gzip-compressed CSV. They retain the case-level outcomes needed
for metric aggregation and paired comparisons, but contain no longitude,
latitude, raw sequence, or runtime column. `case_id` and `analogue_case_id`
are stable pseudonyms rather than source filenames. Evaluation-case IDs follow
the frozen bootstrap case index and are consistent across released tables;
other trajectory IDs are derived with SHA-256.

## Files

- `canonical_cross_method_cases.csv.gz`: case-level, seed-averaged Hit@R90
  outcomes for the seven cross-method rows. The grid row uses the
  deterministic trajectory-identifier tie rule reported in the article.
- `matched_grid_cases.csv.gz`: initial matched full-grid/last-state case matrix
  retained for the saved bootstrap order.
- `tie_sensitivity_cases.csv.gz`: deterministic, inclusive-boundary, and
  top-5/top-10/top-20 matched-grid variants.
- `tie_boundary_cases.csv.gz`: number and prevalence of candidates tied at
  the truncation boundary.
- `matched_familiarity_cases.csv.gz`: matched outcomes joined to the
  training-only close-analogue indicator.
- `catalogue_assignment_sensitivity_cases.csv.gz`: paired full-grid and
  last-state outcomes under the implemented nearest-medoid candidate labels
  and original DBSCAN-component labels; all other design elements are fixed.
- `matched_robustness_evaluation37_cases.csv.gz`: point-count, elapsed-time,
  and cumulative-distance matched predictions for the evaluation cohort.
- `matched_robustness_expanded46_cases.csv.gz`: point-count matched predictions
  for the overlapping expanded sensitivity cohort.
- `geometric_distance_cases.csv.gz`: aligned pointwise, DTW, and discrete
  Fréchet retrieval outcomes.
- `ambiguity_binned_cases.csv.gz`: training-only location-ambiguity features.
- `ambiguity_prediction_cases.csv.gz`: prediction outcomes joined to those
  ambiguity features.

Boolean case columns use `True` and `False`. `covered_r90=False` denotes an
endpoint outside every training-derived R90 destination support; it remains
in the denominator and `hit_r90_all=False`. In the canonical cross-method
file, `hit_r90_all` is instead the case-level mean over fitted seeds and is
therefore numeric from 0 to 1. Distance errors are metres. Observation ratios
are fractions (`0.25`, `0.50`, `0.66`, and `0.75`).

To load a file with pandas:

```python
import pandas as pd

cases = pd.read_csv("reference/predictions/tie_sensitivity_cases.csv.gz")
```
