# Electricity client

- Source: UCI ElectricityLoadDiagrams20112014 (Trindade), hourly consumption of 321 clients for
  2012-2014 as prepared by the LSTNet authors; one client (column 104, the first with no zero
  readings), 26,304 hourly points.
- Mirror: `laiguokun/multivariate-time-series-data` on GitHub, pinned by sha256. Monash
  (zenodo) and GIFT-Eval (Hugging Face) are blocked from cloud sessions (OQ Q4, Q10).
- Units: (origin, horizon) pairs for horizons 1 and 24 hours, built by the baseline helper; the
  embargo between temporal folds is max horizon + max lag = 48 origins.
- Guarantee: forecasting selective risk is `holdout_empirical` only (D15, Q6). The ACI interval
  check (T1-forecast) needs at least 2,000 evaluation steps per horizon; the sealed window has
  about 5,200.
- Loss (provisional, Q5): relative error above 10% with an absolute floor of 1.0.
