# Manuscript claim-to-artifact map

Files under `results/` are compact presentation outputs. Their authoritative
inputs are stored under `reference/`.

| Manuscript component | Config | Implementation | Authoritative result / paper output |
|---|---|---|---|
| GeoLife filtering, modes, cohorts, chronology | `discovery.json`, `evaluation37.json` | `preprocessing.py`, `data.py` | `reference/manifests/`, `protocol_flow.svg` |
| Destination catalogue and Hit@R90 | `evaluation37.json` | `catalogue.py`, `metrics.py` | `cross_method_cases.csv.gz`, `main_hit_r90.csv` |
| Baselines and four representative families | `discovery.json`, `evaluation37.json` | `methods/`, `evaluate.py` | `reference/results/evaluation37/`, `main_hit_r90.csv` |
| Full-grid versus last-state effect | `matched_ablation.json`, `tie_sensitivity.json` | `grid_pattern_retrieval.py`, `matched_ablation.py`, `tie_sensitivity.py` | `matched_robustness_bootstrap.csv`, `matched_grid_contrast.csv`, Figure 2 |
| Paired hierarchical intervals | analysis configs | `bootstrap.py` and analysis-specific vector-preserving routines | `bootstrap_resample_ids.npz`, downstream interval tables |
| Destination and prefix familiarity | `evaluation37.json`, `matched_familiarity.json` | `familiarity.py`, `matched_familiarity.py` | `destination_familiarity.csv`, `matched_prefix_familiarity.csv` |
| Training-only ambiguity and user diagnostics | `ambiguity.json`, `matched_diagnostics.json` | `ambiguity.py`, `diagnostic_statistics.py`, `matched_diagnostics.py` | ambiguity tables, Figure 3 |
| Geometric-distance sensitivity | `geometric_distance.json` | `geometric_distance.py` | geometric-distance tables |
| Boundary ties and top-k sensitivity | `tie_sensitivity.json` | `tie_sensitivity.py` | `tie_stage_sensitivity.csv` |
| Observation, quality, D200, and expanded-cohort sensitivity | `matched_robustness.json`, `expanded46.json` | `matched_robustness.py` | observation-definition, D200, and expanded-cohort tables |

`reference/manifests/run_registry.json` records whether each run was used for
configuration selection, frozen evaluation, post-hoc diagnosis, or
sensitivity analysis.
