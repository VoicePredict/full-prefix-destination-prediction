# External dataset

The artifact setup command installs GeoLife 1.3 here:

```bash
python3 scripts/reproduce.py setup
```

An existing extraction can instead be used in place without copying it:

```bash
make reproduce DATA_ROOT="/path/to/Geolife Trajectories 1.3"
```

The resulting layout is:

```text
data/Geolife/
├── Data/
│   └── 000/Trajectory/*.plt
└── User Guide-1.3.pdf
```

GeoLife is excluded from version control and from the artifact ZIP. The setup
command reuses the supplied root, or downloads the official Microsoft archive
when the default root is absent, and verifies every file against the bundled
SHA-256 manifest.
