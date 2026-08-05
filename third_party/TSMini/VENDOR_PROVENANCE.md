# Vendored TSMini provenance

- Upstream repository: `https://github.com/changyanchuan/TSMini.git`
- Pinned upstream commit: `1fac8436836fc5f03414c9d4fdcec8d2d5a3c4be`
- Vendored content: Python source, upstream readme, requirements, and
  `.gitignore`.
- Excluded content: upstream `.git`, generated data, snapshots, and Python
  caches.

The public encoder and `model/lambdaloss.py` are imported without source
modification. Experiment-specific data adaptation, training orchestration, and
evaluation are implemented once in
`../../src/destination_prediction/methods/tsmini.py`. Its source directory is
added to the import search path only because the unmodified upstream modules
use top-level intra-project imports.
