from pathlib import Path
import time

import pandas as pd
import requests


BASE_URL = "https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json"
FROM_DATE = "2024-01-01"
TILL_DATE = "2026-12-31"
LIMIT = 100
SLEEP_SECONDS = 0.2
OUT_PATH = Path("/Users/maria/Desktop/Code/HSE/COURSEBOOK/data/underlying/imoex_underlying_daily.parquet")


def main() -> int:
    session = requests.Session()
    rows = []
    start = 0

    while True:
        params = {
            "from": FROM_DATE,
            "till": TILL_DATE,
            "interval": 24,
            "iss.meta": "off",
            "start": start,
            "limit": LIMIT,
        }
        print(f"GET start={start}: {BASE_URL}")
        response = session.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        candles = payload.get("candles", {})
        columns = candles.get("columns", [])
        data = candles.get("data", [])
        page = pd.DataFrame(data, columns=columns)

        if page.empty:
            break

        rows.append(page)
        print(f"  rows={len(page)}")

        if len(page) < LIMIT:
            break

        start += LIMIT
        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("No candles downloaded.")
        return 1

    df = pd.concat(rows, ignore_index=True)
    rename_map = {
        "begin": "TRADEDATE",
        "close": "CLOSE",
        "open": "OPEN",
        "high": "HIGH",
        "low": "LOW",
        "volume": "VOLUME",
    }
    df = df.rename(columns=rename_map)

    keep_cols = ["TRADEDATE", "CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"]
    df = df[[col for col in keep_cols if col in df.columns]].copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce").dt.normalize()
    df = df.drop_duplicates(subset=["TRADEDATE"]).sort_values("TRADEDATE").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"Saved to {OUT_PATH}")
    print("shape:", df.shape)
    print("min date:", df["TRADEDATE"].min())
    print("max date:", df["TRADEDATE"].max())
    print("NaN stats:")
    print(df.isna().mean().sort_values(ascending=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
