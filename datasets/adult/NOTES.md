# Adult

- Source: UCI Adult (census income), train + test concatenated: 48,842 rows, 14 inputs, binary
  target `class` (`<=50K` / `>50K`, about 24% positive).
- Mirror: AutoGluon's public S3 copy (`autogluon.s3.amazonaws.com/datasets/Inc/`), pinned by
  sha256. UCI, OpenML and figshare are blocked from cloud sessions (OQ Q4). Alternates checked
  during planning: PMLB (`EpistasisLab/pmlb`, categoricals label-encoded) and
  `jbrownlee/Datasets/adult-all.csv` (raw strings, no header); neither is byte-identical.
- Quirks: `?` marks missing values (6,465 cells, mapped to null); 52 exact duplicate rows, which
  the split keeps in one fold (Q7).
- Role in the suite: mixed types, substantial irreducible noise; expected to be dominated by the
  `irreducible` reason code (HANDOFF 10.1).
- Slices: `sex` and `race` are declared as watch slices for per-group selective risk (OQ Q12). They
  stay usable as inputs unless the owner lists them in `constraints.forbidden_inputs`.
