"""California Housing (1990 census block groups), 20,640 rows, from the Keras GCS copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amx.data.fetch import fetch

URLS = ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/california_housing.npz"]
SHA256 = "1a2e3a52e0398de6463aebe6f4a8da34fb21fbb6b934cf88c3425e766f2a1a6f"
COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]


def load(cache_dir: Path) -> pd.DataFrame:
    path = fetch(URLS, SHA256, Path(cache_dir) / "california_housing" / "california_housing.npz")
    with np.load(path) as z:
        x, y = z["x"], z["y"]
    df = pd.DataFrame(x, columns=COLUMNS)
    df["median_house_value"] = y
    df.insert(0, "row_id", [f"r{i:05d}" for i in range(len(df))])
    return df
