# Commands

`reproduce.py` is the public entry point:

```bash
python3 scripts/reproduce.py setup
python3 scripts/reproduce.py verify
.venv/bin/python scripts/reproduce.py results
.venv/bin/python scripts/reproduce.py full
```

The remaining scripts are focused utilities invoked by that entry point:

- `verify_artifact.py`: integrity, schema, and numerical-claim checks;
- `verify_protocol.py`: frozen evaluation configuration and cohort check;
- `verify_geolife.py`: GeoLife structure and SHA-256 verification;
- `export_paper_tables.py`: paper table generation;
- `generate_figures.py`: SVG figure generation;
- `update_checksums.py`: release-manifest maintenance.
