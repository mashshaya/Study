#!/usr/bin/env python3
"""Shared helpers for the IMOEX options 2024-2026 pipeline."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import re
import time

import pandas as pd
import requests


BASE_URL = "https://iss.moex.com/iss"
SEARCH_URL = f"{BASE_URL}/securities.json"
HISTORY_URL_TEMPLATE = (
    f"{BASE_URL}/history/engines/futures/markets/options/securities/{{secid}}.json"
)

DATE_FROM = "2024-01-01"
DATE_TILL = "2026-12-31"
TIMEOUT = 20
RETRIES = 3
SLEEP_SECONDS = 0.2
SEARCH_LIMIT = 100
HISTORY_LIMIT = 100

PIPELINE_DIR = Path("data/imoex_options_2024_2026")

CANDIDATES_PARQUET = PIPELINE_DIR / "historical_secids_candidates.parquet"
CANDIDATES_CSV = PIPELINE_DIR / "historical_secids_candidates.csv"
DISCOVERY_REPORT = PIPELINE_DIR / "discovery_report.txt"

VALIDATED_PARQUET = PIPELINE_DIR / "validated_historical_secids.parquet"
VALIDATED_CSV = PIPELINE_DIR / "validated_historical_secids.csv"
VALIDATION_REPORT = PIPELINE_DIR / "validation_report.txt"

FINAL_HISTORY_PARQUET = PIPELINE_DIR / "imoex_options_daily_history_2024_2026.parquet"
FINAL_HISTORY_CSV = PIPELINE_DIR / "imoex_options_daily_history_2024_2026.csv"
DOWNLOAD_REPORT = PIPELINE_DIR / "download_report.txt"
QUALITY_SUMMARY_JSON = PIPELINE_DIR / "quality_summary.json"
COVERAGE_BY_YEAR_CSV = PIPELINE_DIR / "coverage_by_year.csv"
ROWS_BY_SECID_CSV = PIPELINE_DIR / "rows_by_secid.csv"
MISSING_SHARE_CSV = PIPELINE_DIR / "missing_share.csv"

DISCOVERY_BASE_QUERIES = [
    "IMOEXP",
    "IMOEX option",
    "IMOEX",
    "MIX",
    "MXI",
    "Прем. европ.",
    "Нед. прем. европ.",
    "Прем. европ. Call",
    "Прем. европ. Put",
]

DISCOVERY_TEXT_HINTS = ["imoex", "imoexp", "mix", "mxi", "индекс мосбиржи", "мосбир"]
OPTION_HINTS = [
    "option",
    "опцион",
    "call",
    "put",
    "прем.",
    "нед. прем.",
    "марж.",
    "futures_options",
]
EXCLUDE_HINTS = [
    "bond",
    "share",
    "stock",
    "etf",
    "акция",
    "облигац",
    "fund",
]

FINAL_RAW_COLUMNS = [
    "SECID",
    "TRADEDATE",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "SETTLEPRICE",
    "VALUE",
    "VOLUME",
    "OPENPOSITION",
    "NUMTRADES",
]


def ensure_output_dir() -> None:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    return logger


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "imoex-options-pipeline/1.0 (requests/pandas research script)"}
    )
    return session


def build_url(url: str, params: dict[str, object]) -> str:
    request = requests.PreparedRequest()
    request.prepare_url(url, params)
    return request.url


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_lower(value: object) -> str:
    return safe_text(value).lower()


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    *,
    logger: logging.Logger,
    context: str,
) -> tuple[dict | None, str | None]:
    full_url = build_url(url, params)
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            logger.info("GET %s attempt %s/%s: %s", context, attempt, RETRIES, full_url)
            response = session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json(), None
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "%s failed on attempt %s/%s: %s", context, attempt, RETRIES, exc
            )
            if attempt < RETRIES:
                time.sleep(SLEEP_SECONDS)

    return None, str(last_error)


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


def fetch_search_results(
    session: requests.Session,
    query: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None]:
    params = {"iss.meta": "off", "q": query, "limit": SEARCH_LIMIT}
    payload, error = fetch_json(
        session,
        SEARCH_URL,
        params,
        logger=logger,
        context=f"search:{query}",
    )
    if payload is None:
        return pd.DataFrame(), error
    return table_to_frame(payload, "securities"), None


def fetch_full_history(
    session: requests.Session,
    secid: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None]:
    url = HISTORY_URL_TEMPLATE.format(secid=secid)
    params = {
        "from": DATE_FROM,
        "till": DATE_TILL,
        "iss.meta": "off",
        "start": 0,
        "limit": HISTORY_LIMIT,
    }
    frames: list[pd.DataFrame] = []
    columns: list[str] = []

    while True:
        payload, error = fetch_json(
            session,
            url,
            params,
            logger=logger,
            context=f"history:{secid}:start={params['start']}",
        )
        if payload is None:
            return pd.DataFrame(columns=columns), error

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

    combined = pd.concat(frames, ignore_index=True)
    return combined, None


def is_option_like_row(row: pd.Series) -> bool:
    blob = " | ".join(
        safe_lower(row.get(column))
        for column in ["secid", "shortname", "name", "type", "group", "_query", "_strategy"]
    )
    include = any(token in blob for token in DISCOVERY_TEXT_HINTS)
    option_like = any(token in blob for token in OPTION_HINTS)
    excluded = any(
        token in safe_lower(row.get("type")) or token in safe_lower(row.get("group"))
        for token in EXCLUDE_HINTS
    )
    return include and option_like and not excluded


def generate_secid_variants(secid: str) -> list[str]:
    value = safe_text(secid)
    variants = [value]
    match = re.search(r"(\d)([A-Z]?)$", value)
    if match:
        start, end = match.span(1)
        for year_digit in ["4", "5", "6", "7", "8"]:
            variants.append(value[:start] + year_digit + value[end:])
    return [variant for variant in dict.fromkeys(variants) if variant]


def generate_shortname_variants(shortname: str) -> list[str]:
    value = safe_text(shortname)
    variants = [value]
    match = re.search(r"(\d{4})(\d{2})(C[AE]|P[AE])", value)
    if match:
        ddmm = match.group(1)
        suffix = match.group(3)
        for yy in ["24", "25", "26", "27", "28"]:
            strike_match = re.search(r"(C[AE]|P[AE])(.+)$", value)
            strike = strike_match.group(2) if strike_match else ""
            variants.append(f"IMOEXP{ddmm}{yy}{suffix}{strike}")
    return [variant for variant in dict.fromkeys(variants) if variant]


def generate_queries_from_row(row: pd.Series) -> list[tuple[str, str]]:
    secid = safe_text(row.get("secid"))
    shortname = safe_text(row.get("shortname"))
    name = safe_text(row.get("name"))
    queries: list[tuple[str, str]] = []

    for variant in generate_secid_variants(secid):
        queries.append(("secid_variant", variant))
    for variant in generate_shortname_variants(shortname):
        queries.append(("shortname_variant", variant))

    if shortname:
        prefix_match = re.match(r"([A-Z]+)", shortname)
        if prefix_match:
            queries.append(("shortname_prefix", prefix_match.group(1)))

    if name:
        name_match = re.split(r"с исп\\.|на IMOEX", name)
        if name_match:
            fragment = safe_text(name_match[0])
            if fragment:
                queries.append(("name_fragment", fragment))

    return list(dict.fromkeys(queries))


def parse_option_metadata_from_row(row: pd.Series) -> tuple[str | None, float | None, str | None]:
    shortname = safe_text(row.get("shortname"))
    name = safe_text(row.get("name"))
    secid = safe_text(row.get("secid"))

    option_type: str | None = None
    if any(token in shortname for token in ["CE", "CA"]) or "call" in name.lower():
        option_type = "C"
    if any(token in shortname for token in ["PE", "PA"]) or "put" in name.lower():
        option_type = "P"

    strike: float | None = None
    strike_match = re.search(r"(?:CE|PE|CA|PA)(\d+(?:\.\d+)?)$", shortname)
    if not strike_match:
        strike_match = re.search(r"(?:Call|Put)\s+(\d+(?:[.,]\d+)?)", name, re.IGNORECASE)
    if not strike_match:
        strike_match = re.search(r"([CP][A-Z])(\d+(?:\.\d+)?)$", secid)
    if strike_match:
        strike_value = strike_match.group(1 if len(strike_match.groups()) == 1 else 2)
        strike = float(strike_value.replace(",", "."))

    expiry_date: str | None = None
    expiry_match = re.search(r"(\d{2})(\d{2})(\d{2})(?:CE|PE|CA|PA)", shortname)
    if expiry_match:
        day, month, year = expiry_match.groups()
        expiry_date = f"20{year}-{month}-{day}"

    return option_type, strike, expiry_date


def add_parsed_contract_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["option_type"] = pd.Series(dtype="object")
        result["strike"] = pd.Series(dtype="float64")
        result["expiry_date"] = pd.Series(dtype="object")
        return result

    option_types: list[str | None] = []
    strikes: list[float | None] = []
    expiry_dates: list[str | None] = []

    for _, row in df.iterrows():
        option_type, strike, expiry_date = parse_option_metadata_from_row(row)
        option_types.append(option_type)
        strikes.append(strike)
        expiry_dates.append(expiry_date)

    result = df.copy()
    result["option_type"] = option_types
    result["strike"] = strikes
    result["expiry_date"] = expiry_dates
    return result


def summarize_history_frame(df: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": int(len(df)),
        "first_tradedate": None,
        "last_tradedate": None,
    }
    if df.empty or "TRADEDATE" not in df.columns:
        return summary
    dates = pd.to_datetime(df["TRADEDATE"], errors="coerce").dropna()
    if not dates.empty:
        summary["first_tradedate"] = str(dates.min().date())
        summary["last_tradedate"] = str(dates.max().date())
    return summary


def write_text_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_dataframe(df: pd.DataFrame, parquet_path: Path, csv_path: Path | None = None) -> None:
    frame = df.copy()
    if frame.empty and not list(frame.columns):
        frame = pd.DataFrame({"note": []})
    frame.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        frame.to_csv(csv_path, index=False)
