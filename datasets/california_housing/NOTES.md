# California Housing

- Source: Pace & Barry (1997), 1990 census block groups; 20,640 rows, 8 numeric inputs, target
  `median_house_value` in dollars.
- Mirror: the Keras copy on Google Cloud Storage (`tf-keras-datasets/california_housing.npz`),
  pinned by sha256; raw units, no missing values. scikit-learn's fetcher uses figshare, which is
  blocked from cloud sessions (OQ Q4). Alternate: `ageron/handson-ml2` `housing.csv` (same rows,
  207 missing `total_bedrooms`, extra categorical column), not byte-identical.
- Quirk: the target is top-coded at 500,001 (965 rows, 4.7%). Per OQ Q5 the default keeps the
  labels as they are, reports capped rows as a slice, and states "risk measured against top-coded
  gold". The profiler must flag the pile-up (HANDOFF 13).
- Loss (provisional, Q5): `err_gt_tol` with relative tolerance 0.25.
