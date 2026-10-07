# Validation datasets

One directory per dataset: `spec.yaml` (the TaskSpec), `loader.py` (downloads pinned mirrors into
`~/.cache/amx/datasets` and returns a table), `NOTES.md`, `LICENSE_NOTE.md`. Data is never
committed. Dataset and column names live only here, never in `src/` (domain-blind lint).

Loss parameters and bands in these specs are provisional defaults from `docs/OQ.md` Q5 until the
owner confirms them. Split seeds here are benchmark seeds (OQ Q3), never reused for a release.
