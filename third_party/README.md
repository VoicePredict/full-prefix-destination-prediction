# Third-party code

The experiments use the public TSMini encoder at the exact upstream commit
recorded under `TSMini/`. Its source is not copied into this artifact.
`python3 scripts/reproduce.py setup` downloads that revision into
`third_party/TSMini/`; discovery and both evaluation cohorts import this same
pinned copy. The evaluation configuration and original cohort freeze are
checked independently by `scripts/verify_protocol.py`.
