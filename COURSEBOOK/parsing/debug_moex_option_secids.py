#!/usr/bin/env python3
"""Debug whether candidate MOEX option SECIDs have history in wider date ranges."""

from pathlib import Path
import json
import logging
import time

import pandas as pd
import requests


BASE_URL = "https://iss.moex.com/iss/history/engines/futures/markets/options/securities/{secid}.json"
CANDIDATES_PATH = Path("data/raw/moex_options_2y_candidates.parquet")
REPORT_PATH = Path("data/raw/moex_option_secids_debug_report.txt")
TIMEOUT = 20
RETRIES = 3
PAGE_LIMIT = 100
SLEEP_SECONDS = 0.2

DATE_RANGES = [
    ("2023-01-01", "2024-12-31"),
    ("2024-01-01", "2026-05-20"),
    ("2015-01-01", "2026-05-20"),
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("debug_moex_option_secids")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    return logger


def build_url(url: str, params: dict[str, object]) -> str:
    request = requests.PreparedRequest()
    request.prepare_url(url, params)
    return request.url


def table_to_frame(payload: dict | None, table_name: str) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    block = payload.get(table_name) or {}
    columns = block.get("columns") or []
    data = block.get("data") or []
    return pd.DataFrame(data, columns=columns)


def cursor_to_dict(payload: dict | None, table_name: str) -> dict[str, object]:
    if not payload:
        return {}
    block = payload.get(f"{table_name}.cursor")
    if not isinstance(block, dict):
        return {}
    columns = block.get("columns") or []
    data = block.get("data") or []
    if not columns or not data:
        return {}
    return dict(zip(columns, data[0]))


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    *,
    logger: logging.Logger,
    context: str,
) -> dict | None:
    full_url = build_url(url, params)
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            logger.info("GET %s attempt %s/%s: %s", context, attempt, RETRIES, full_url)
            response = session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("%s failed on attempt %s/%s: %s", context, attempt, RETRIES, exc)
            if attempt < RETRIES:
                time.sleep(SLEEP_SECONDS)
    logger.error("%s failed after %s attempts: %s", context, RETRIES, last_error)
    return None


def fetch_history(
    session: requests.Session,
    secid: str,
    date_from: str,
    date_till: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None]:
    url = BASE_URL.format(secid=secid)
    params = {
        "from": date_from,
        "till": date_till,
        "iss.meta": "off",
        "start": 0,
        "limit": PAGE_LIMIT,
    }
    frames: list[pd.DataFrame] = []
    columns: list[str] = []

    while True:
        payload = fetch_json(
            session,
            url,
            params,
            logger=logger,
            context=f"{secid}:{date_from}:{date_till}:start={params['start']}",
        )
        if payload is None:
            return pd.DataFrame(columns=columns), "request failed"

        frame = table_to_frame(payload, "history")
        if list(frame.columns) and not columns:
            columns = frame.columns.tolist()
        if frame.empty:
            break

        frames.append(frame)
        cursor = cursor_to_dict(payload, "history")
        total = cursor.get("TOTAL")
        page_size = cursor.get("PAGESIZE")
        if total is None or page_size is None:
            break

        try:
            total_int = int(total)
            page_size_int = int(page_size)
        except (TypeError, ValueError):
            break

        next_start = int(params["start"]) + len(frame)
        if next_start >= total_int or len(frame) < page_size_int:
            break

        params["start"] = next_start
        time.sleep(SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame(columns=columns), None

    return pd.concat(frames, ignore_index=True), None


def nonzero_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns or df.empty:
        return 0
    series = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return int((series != 0).sum())


def summarize_frame(df: pd.DataFrame) -> dict[str, object]:
    summary = {
        "total_rows": int(len(df)),
        "first_tradedate": None,
        "last_tradedate": None,
        "nonzero_settleprice_count": 0,
        "nonzero_volume_count": 0,
        "first_5_rows": [],
    }
    if df.empty:
        return summary

    working = df.copy()
    if "TRADEDATE" in working.columns:
        working["TRADEDATE"] = pd.to_datetime(working["TRADEDATE"], errors="coerce")
        valid_dates = working["TRADEDATE"].dropna()
        if not valid_dates.empty:
            summary["first_tradedate"] = str(valid_dates.min().date())
            summary["last_tradedate"] = str(valid_dates.max().date())

    summary["nonzero_settleprice_count"] = nonzero_count(working, "SETTLEPRICE")
    summary["nonzero_volume_count"] = nonzero_count(working, "VOLUME")
    sample = working.head(5).copy()
    for column in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[column]):
            sample[column] = sample[column].astype(str)
    summary["first_5_rows"] = sample.to_dict(orient="records")
    return summary


def main() -> int:
    logger = setup_logger()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(CANDIDATES_PATH)
    secids = candidates["secid"].dropna().astype(str).head(50).tolist()

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "debug_moex_option_secids.py (requests/pandas debug script)"}
    )

    lines = [
        "MOEX option SECID debug report",
        f"candidate source: {CANDIDATES_PATH}",
        f"tested SECIDs: {len(secids)}",
        "",
    ]

    for secid in secids:
        logger.info("Checking SECID %s", secid)
        lines.append(f"SECID: {secid}")
        for date_from, date_till in DATE_RANGES:
            df, error = fetch_history(session, secid, date_from, date_till, logger)
            summary = summarize_frame(df)
            lines.append(f"  range: {date_from} -> {date_till}")
            if error:
                lines.append(f"    error: {error}")
            lines.append(f"    total_rows: {summary['total_rows']}")
            lines.append(f"    first_TRADEDATE: {summary['first_tradedate']}")
            lines.append(f"    last_TRADEDATE: {summary['last_tradedate']}")
            lines.append(
                f"    nonzero_SETTLEPRICE_count: {summary['nonzero_settleprice_count']}"
            )
            lines.append(f"    nonzero_VOLUME_count: {summary['nonzero_volume_count']}")
            if summary["first_5_rows"]:
                lines.append("    first_5_rows:")
                for row in summary["first_5_rows"]:
                    lines.append(
                        "      " + json.dumps(row, ensure_ascii=False, default=str)
                    )
            else:
                lines.append("    first_5_rows: none")
            time.sleep(SLEEP_SECONDS)
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report saved to %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
