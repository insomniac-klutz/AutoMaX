"""Adult (UCI census income), 48,842 rows: the AutoGluon S3 copy of the UCI train+test files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amx.data.fetch import fetch

FILES = {
    "train.csv": (
        ["https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv"],
        "0df92db5f2c63465df6222af460381b0e8d215988522f6b02663cf73d5a4b4b3",
    ),
    "test.csv": (
        ["https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv"],
        "8627c8224d9377a7e3a755d5a5bd4801931dd034c3f535c1b4ad0f07a10a5e29",
    ),
}


def load(cache_dir: Path) -> pd.DataFrame:
    root = Path(cache_dir) / "adult"
    parts = []
    for name, (urls, sha) in FILES.items():
        path = fetch(urls, sha, root / name)
        parts.append(pd.read_csv(path, skipinitialspace=True, na_values=["?"]))
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row_id", [f"r{i:05d}" for i in range(len(df))])
    return df
