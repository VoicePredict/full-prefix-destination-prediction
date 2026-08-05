# External dataset

The artifact setup command installs GeoLife 1.3 here:

```bash
python3 scripts/reproduce.py setup
```

The resulting layout is:

```text
data/Geolife/
├── Data/
│   └── 000/Trajectory/*.plt
└── User Guide-1.3.pdf
```

GeoLife is excluded from version control and from the artifact ZIP. The setup
command downloads the official Microsoft archive, extracts it, and verifies
every file against the bundled SHA-256 manifest.
