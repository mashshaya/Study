from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "final_dataset_29_30_32.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(load_dataset().head().to_string(index=False))
