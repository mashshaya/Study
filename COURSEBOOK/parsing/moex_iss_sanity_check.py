#!/usr/bin/env python3
"""Small MOEX ISS sanity-check script."""

from pathlib import Path
import json
import logging

import pandas as pd
import requests


REPORT_PATH = Path("data/raw/moex_iss_sanity_check_report.txt")
TIMEOUT = 20


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("moex_iss_sanity_check")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    return logger


def table_to_frame(payload: dict, table_name: str) -> pd.DataFrame:
    block = payload.get(table_name) or {}
    columns = block.get("columns") or []
    data = block.get("data") or []
    return pd.DataFrame(data, columns=columns)


def format_frame(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "<empty>"
    return df.head(max_rows).to_string(index=False)


def run_check(
    name: str,
    url: str,
    params: dict,
    lines: list[str],
    logger: logging.Logger,
) -> None:
    lines.append(f"=== {name} ===")
    lines.append(f"URL: {url}")
    lines.append(f"Params: {json.dumps(params, ensure_ascii=False, sort_keys=True)}")

    try:
        logger.info("Running %s", name)
        response = requests.get(url, params=params, timeout=TIMEOUT)
        lines.append(f"Status code: {response.status_code}")
        response.raise_for_status()
        payload = response.json()

        if name == "API availability check":
            lines.append(f"Top-level JSON keys: {list(payload.keys())}")

        elif name == "Search check":
            lines.append(f"Table names: {list(payload.keys())}")
            df = table_to_frame(payload, "securities")
            lines.append(f"Securities columns: {list(df.columns)}")
            lines.append("First rows:")
            lines.append(format_frame(df, max_rows=10))

        elif name == "Known index candles check":
            df = table_to_frame(payload, "candles")
            lines.append(f"Candles columns: {list(df.columns)}")
            lines.append("Returned rows:")
            lines.append(format_frame(df, max_rows=10))

        elif name == "Known options history endpoint check":
            lines.append(f"Table names: {list(payload.keys())}")
            df = table_to_frame(payload, "history")
            lines.append(f"History columns: {list(df.columns)}")
            lines.append("Returned rows:")
            lines.append(format_frame(df, max_rows=20))

    except Exception as exc:
        message = f"ERROR: {type(exc).__name__}: {exc}"
        logger.error("%s failed: %s", name, message)
        lines.append(message)

    lines.append("")


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    lines: list[str] = []

    checks = [
        (
            "API availability check",
            "https://iss.moex.com/iss/index.json",
            {"iss.meta": "off"},
        ),
        (
            "Search check",
            "https://iss.moex.com/iss/securities.json",
            {"iss.meta": "off", "q": "IMOEX", "limit": 10},
        ),
        (
            "Known index candles check",
            "https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json",
            {
                "from": "2024-01-03",
                "till": "2024-01-03",
                "interval": 24,
                "iss.meta": "off",
            },
        ),
        (
            "Known options history endpoint check",
            "https://iss.moex.com/iss/history/engines/futures/markets/options/securities/AF39CE6.json",
            {
                "from": "2026-05-01",
                "till": "2026-05-19",
                "iss.meta": "off",
            },
        ),
    ]

    for name, url, params in checks:
        run_check(name, url, params, lines, logger)

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Report saved to %s", REPORT_PATH)
    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
