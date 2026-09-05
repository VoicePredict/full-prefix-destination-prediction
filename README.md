# When Does Full-Prefix Geometry Matter?

Reproducibility artifact for *When Does Full-Prefix Geometry Matter for
Personal Destination Prediction? A Leakage-Controlled Multi-Ratio
Evaluation*.

## Reviewer path

**1. Fast verification of the frozen results** (no download or third-party
packages):

```bash
python3 scripts/reproduce.py verify
```

This checks release integrity, case schemas, cohort allocation, and every
number in `results/expected_claims.json`.

**2. Regenerate the article tables and figures** from the bundled validated
results:

```bash
make environment
make results
```

**3. Repeat every experiment** from the original sources:

```bash
make reproduce
```

Python 3.10 and GNU Make are required. If Python 3.10 is not exposed as
`python3`, pass its executable explicitly, for example
`make reproduce PYTHON=python3.10`.
The full path also requires the Python `venv` module, outbound HTTPS access to
Microsoft, GitHub, PyPI, and the PyTorch CPU index, and several gigabytes of
free disk space.

`requirements.txt` is the install list and pins the direct scientific
dependencies. `constraints.txt` installs nothing by itself; it fixes the
transitive versions selected while that list is installed. The same direct
dependencies appear in `pyproject.toml` as package metadata.

If GeoLife 1.3 is already available elsewhere, no copy is required:

```bash
make reproduce DATA_ROOT="/path/to/Geolife Trajectories 1.3"
```

This is the computationally intensive path. It first verifies the submitted
files, creates the pinned environment, obtains and verifies the external
inputs, runs the dependency graph, validates every stage, builds a fresh
normalized result bundle, regenerates the paper tables and figures from that
bundle, and fails if their published values differ from the validated files
in `results/`.

External inputs can also be prepared separately:

```bash
python3 scripts/reproduce.py setup
```

This reuses GeoLife when `--data-root PATH` is supplied, or otherwise downloads
the official GeoLife 1.3 ZIP to a temporary cache and extracts it to
`data/Geolife/`. It also downloads the pinned TSMini revision under
`third_party/TSMini/` and verifies both external inputs. Repeating the command
is safe; installed inputs are detected and reused.

## Structure

| Path | Purpose |
|---|---|
| `src/destination_prediction/` | Scientific implementation, separated into preprocessing, catalogue, geometry, metrics, bootstrap, methods, analyses, and validation |
| `configs/` | Frozen configuration for each scientific run |
| `scripts/` | One public runner and small reporting/verification utilities |
| `reference/` | Three-part bundle: manifests, coordinate-free predictions, and validated results |
| `results/` | Paper-facing tables and figures |
| `tests/` | Scientific invariants, artifact integrity, and fresh-bundle equivalence checks |
| `docs/` | Dataset details, reproduction protocol, and claim-to-file map |
| `data/` | Downloaded GeoLife data; ignored by Git |
| `third_party/` | Downloaded pinned TSMini source; ignored by Git |
| `work/` | Generated runs, logs, and checkpoints; ignored by Git |

After a complete run, the independently regenerated public bundle is written
to `work/reproduced/` and its paper outputs to `work/reproduced-paper/`.

Three detailed guides are retained under `docs/`:
[`DATA.md`](docs/DATA.md) defines the dataset boundary and preprocessing,
[`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) describes the execution graph
and deterministic decisions, and [`CLAIM_MAP.md`](docs/CLAIM_MAP.md) maps
manuscript claims to configuration, implementation, and results.

## Experiment organization

The complete workflow consists of ten registered scientific runs. They share
one implementation under `src/`; no experiment-specific code trees are
duplicated. `scripts/reproduce.py full` resolves their dependency graph,
records portable manifests, validates every stage, and resumes validated
stages when their code and configuration fingerprints are unchanged.

Configurations were selected on a separate 30-user discovery cohort and then
fixed for evaluation on 37 other users. The matched last-state grid ablation
is recorded as a post-hoc diagnostic. The overlapping 46-user analysis is
recorded as a sensitivity cohort rather than an independent replication.

The catalogue audit separates DBSCAN region construction from downstream
nearest-medoid labelling and records the 66 of 4,370 training endpoints for
which those assignments differ. `matched_robustness` repeats the matched-grid
calculation with original DBSCAN membership as a sensitivity analysis.

Each workflow receives a `RunContext` containing its configuration, data
root, output directory, and dependency outputs. Dependencies are addressed
only by the stable IDs in `reference/manifests/run_registry.json`.

The registered order and dependencies can be inspected without data access:

```bash
python3 scripts/reproduce.py list
```

## Data boundary

GeoLife and TSMini remain governed by their respective upstream terms. The
dataset is not redistributed; the setup command obtains it from the official
[Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=52367).
TSMini is obtained from its [upstream repository](https://github.com/changyanchuan/TSMini)
at the commit recorded in `third_party/TSMini/VENDOR_PROVENANCE.md`.
