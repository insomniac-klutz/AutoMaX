"""One client of UCI ElectricityLoadDiagrams20112014, hourly 2012-2014 (26,304 points).

Built into (origin, horizon) forecast units with the A0 baseline helper. Every feature uses only
values at or before the origin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from amx.baseline.trivial import make_forecast_units
from amx.data.fetch import fetch

URLS = [
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/"
    "electricity/electricity.txt.gz"
]
SHA256 = "3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa"
CLIENT = 104  # first client column with no zero readings
HORIZONS = (1, 24)
SEASON = 24
MAX_LAG = 24


def load(cache_dir: Path) -> pd.DataFrame:
    path = fetch(URLS, SHA256, Path(cache_dir) / "electricity" / "electricity.txt.gz")
    values = np.loadtxt(path, delimiter=",")[:, CLIENT]
    times = pd.date_range("2012-01-01", periods=values.shape[0], freq="h")
    units = make_forecast_units(values, times, HORIZONS, SEASON, MAX_LAG)
    return units
