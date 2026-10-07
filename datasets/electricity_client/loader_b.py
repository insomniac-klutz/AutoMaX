"""Diagnostic B (pre-registered 2026-10-07, before it was run): a different client, column 127.

Gate item 4 found no certifiable band for column 104 at the provisional Q5 bands {5%, 10%}
(no validity failure; the seasonal-naive baseline is too weak). Column 127 has a calibration
window never examined, so certifying it at the looser, pre-registered bands {20%, 30%} gives a
non-vacuous check of the forecasting path. It does not replace the gate result for column 104.
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
CLIENT = 127  # second client column with no zero readings
HORIZONS = (1, 24)
SEASON = 24
MAX_LAG = 24


def load(cache_dir: Path) -> pd.DataFrame:
    path = fetch(URLS, SHA256, Path(cache_dir) / "electricity" / "electricity.txt.gz")
    values = np.loadtxt(path, delimiter=",")[:, CLIENT]
    times = pd.date_range("2012-01-01", periods=values.shape[0], freq="h")
    return make_forecast_units(values, times, HORIZONS, SEASON, MAX_LAG)
