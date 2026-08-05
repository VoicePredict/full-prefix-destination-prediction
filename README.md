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
python3 scripts/reproduce.py results
```

**3. Repeat every experiment** from the original sources:

```bash
make reproduce
```

This is the computationally intensive path. It downloads and verifies the
external inputs, creates the pinned environment, runs the dependency graph,
and validates every stage.

External inputs can also be prepared separately:

```bash
python3 scripts/reproduce.py setup
```

This downloads the official GeoLife 1.3 ZIP to a temporary cache, extracts it
to `data/Geolife/`, downloads the pinned TSMini revision, installs it under
`third_party/TSMini/`, and verifies the GeoLife file manifest. Repeating the
command is safe; installed inputs are detected and reused.

## Structure

| Path | Purpose |
|---|---|
| `src/destination_prediction/` | Scientific implementation, separated into preprocessing, catalogue, geometry, metrics, bootstrap, methods, analyses, and validation |
| `configs/` | Frozen configuration for each scientific run |
| `scripts/` | One public runner and small reporting/verification utilities |
| `reference/` | Three-part bundle: manifests, coordinate-free predictions, and validated results |
| `results/` | Paper-facing tables and figures |
| `data/` | Downloaded GeoLife data; ignored by Git |
| `third_party/` | Downloaded pinned TSMini source; ignored by Git |
| `work/` | Generated runs, logs, and checkpoints; ignored by Git |

The reviewer workflow is described in
[`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md). The mapping from manuscript
claims to files is in [`docs/CLAIM_MAP.md`](docs/CLAIM_MAP.md).

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

Each workflow receives a `RunContext` containing its configuration, data
root, output directory, and dependency outputs. Dependencies are addressed
only by the stable IDs in `reference/manifests/run_registry.json`.

## Data boundary

GeoLife and TSMini remain governed by their respective upstream terms. The
dataset is not redistributed; the setup command obtains it from the official
[Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=52367).
TSMini is obtained from its [upstream repository](https://github.com/changyanchuan/TSMini)
at the commit recorded in `third_party/TSMini/VENDOR_PROVENANCE.md`.
