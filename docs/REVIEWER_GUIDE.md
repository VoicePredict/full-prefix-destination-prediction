# Reviewer guide

## 1. Verify the submitted artifact

Requirements: Python 3.10 or newer. No third-party packages, raw data, or
network access are needed.

```bash
python3 scripts/reproduce.py verify
```

Successful verification ends with `"status": "passed"` and an empty
`failures` list. The command checks repository hashes, public prediction
schemas, dataset and cohort counts, paired-case completeness, and every value
listed in `results/expected_claims.json`.

## 2. Regenerate paper outputs

```bash
make environment
make results
```

Tables are written to `results/paper_tables/`; figures are written to
`results/figures/`. This step uses only the bundled coordinate-free records
and validated statistics.

## 3. Reproduce all experiments

A complete reproduction from a clean clone is started with one command:

```bash
make reproduce
```

It performs the following operations in order:

1. downloads and verifies GeoLife 1.3;
2. downloads the pinned TSMini revision;
3. creates the pinned Python 3.10 environment;
4. executes configuration selection, both evaluation cohorts, and all
   reported diagnostic and sensitivity analyses;
5. validates every run.

To separate setup from execution, use:

```bash
python3 scripts/reproduce.py setup
make environment
make full
```

The full workflow is computationally intensive because five fixed seeds are
fitted for each neural family in the main cohorts. The reference execution
used Linux x86-64, Python 3.10.12, CPU-only PyTorch, and at least 16 GiB RAM.

## Selected operations

```bash
# Inspect the DAG without data access or downloads
python3 scripts/reproduce.py full --dry-run

# Install only one external component
python3 scripts/reproduce.py setup --component geolife
python3 scripts/reproduce.py setup --component tsmini

# Run one result family and all of its dependencies
.venv/bin/python scripts/reproduce.py full --target matched_robustness

# Use external data and output locations
.venv/bin/python scripts/reproduce.py full \
  --data-root /path/to/Geolife \
  --output-root /path/to/run-output
```

All supplied paths are converted to `pathlib.Path` objects. Runtime manifests
store repository-relative locations or `<external>/<name>`, never absolute
host paths.
