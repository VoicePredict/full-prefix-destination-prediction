# Manifests and provenance

This directory fixes the exact dataset version, preprocessing outcome, cohort
allocation, chronological split, scientific role of each run, and reference
software environment.

The GeoLife source archive contains 182 user directories, 18,670 PLT files,
69 label files, and the Version 1.3 user guide. The
`geolife_1.3_files.sha256` manifest contains all 18,740 files and is verified
by `scripts/verify_geolife.py` after download.

The cohort and preprocessing records contain no GPS coordinates or resampled
sequences. `cohort_manifest.csv` is the single user-allocation record and
states eligibility, discovery membership, prior use, evaluation membership,
expanded-cohort membership, and any exclusion reason. `run_registry.json`
distinguishes configuration selection, frozen evaluation, post-hoc
diagnostics, sensitivity analyses, and the overlapping expanded cohort.
