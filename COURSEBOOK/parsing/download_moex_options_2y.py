#!/usr/bin/env python3
"""Download a 2-year MVP dataset for MOEX IMOEX-related options on futures."""

from pathlib import Path
import json
import logging
import time

import pandas as pd
import requests


BASE_URL = "https://iss.moex.com/iss"
DATE_FROM = "2023-01-01"
DATE_TILL = "2024-12-31"
TIMEOUT = 20
RETRIES = 3
SLEEP_SECONDS = 0.2
DISCOVERY_LIMIT = 100
HISTORY_LIMIT = 100
MAX_CANDIDATE_SECIDS = 50
DISCOVERY_QUERIES = ["IMOEX", "MIX", "MX"]

OUTPUT_DIR = Path("data/raw")
RAW_METADATA_PATH = OUTPUT_DIR / "moex_options_2y_metadata_raw.parquet"
CANDIDATES_PATH = OUTPUT_DIR / "moex_options_2y_candidates.parquet"
HISTORY_PARQUET_PATH = OUTPUT_DIR / "moex_options_2y_history.parquet"
HISTORY_CSV_PATH = OUTPUT_DIR / "moex_options_2y_history.csv"
CANDLES_PARQUET_PATH = OUTPUT_DIR / "moex_imoex_2y_candles.parquet"
CANDLES_CSV_PATH = OUTPUT_DIR / "moex_imoex_2y_candles.csv"
REPORT_PATH = OUTPUT_DIR / "moex_options_2y_report.txt"

SEARCH_URL = f"{BASE_URL}/securities.json"
HISTORY_URL_TEMPLATE = (
    f"{BASE_URL}/history/engines/futures/markets/options/securities/{{secid}}.json"
)
CANDLES_URL = (
    f"{BASE_URL}/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json"
)

TEXT_INCLUDE_HINTS = [
    "imoex",
    "mix",
    "mx",
    "moex",
    "индекс мосбиржи",
]
OPTION_HINTS = [
    "option",
    "options",
    "opt",
    "call",
    "put",
    "опцион",
    "марж.",
    "прем.",
]
OPTION_EXACT_HINTS = ["c", "p"]
EXCLUDE_HINTS = [
    "bond",
    "etf",
    "share",
    "stock",
    "акция",
    "облигац",
    "etf",
]
EXPECTED_HISTORY_COLUMNS = [
    "BOARDID",
    "TRADEDATE",
    "SECID",
    "OPEN",
    "LOW",
    "HIGH",
    "CLOSE",
    "OPENPOSITIONVALUE",
    "VALUE",
    "VOLUME",
    "OPENPOSITION",
    "SETTLEPRICE",
    "WAPRICE",
    "CHANGE",
    "QTY",
    "NUMTRADES",
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("download_moex_options_2y")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    return logger


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


def fetch_discovery_rows(
    session: requests.Session,
    logger: logging.Logger,
    diagnostics: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    urls: list[str] = []

    for query in DISCOVERY_QUERIES:
        params = {"iss.meta": "off", "q": query, "limit": DISCOVERY_LIMIT}
        urls.append(build_url(SEARCH_URL, params))
        payload = fetch_json(
            session,
            SEARCH_URL,
            params,
            logger=logger,
            context=f"discovery:{query}",
        )
        if payload is None:
            diagnostics.append(f"discovery query {query} failed")
            continue

        frame = table_to_frame(payload, "securities")
        logger.info("Discovery query %s returned %s rows", query, len(frame))
        diagnostics.append(f"discovery query {query}: {len(frame)} rows")
        if frame.empty:
            continue

        frame = frame.copy()
        frame["_query"] = query
        frames.append(frame)
        time.sleep(SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame(), urls

    combined = pd.concat(frames, ignore_index=True)
    dedupe_columns = [
        column
        for column in ["secid", "primary_boardid", "type", "group", "name"]
        if column in combined.columns
    ]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    else:
        combined = combined.drop_duplicates().reset_index(drop=True)

    return combined, urls


def candidate_reason(row: pd.Series) -> tuple[bool, str]:
    text_fields = [
        "secid",
        "shortname",
        "name",
        "group",
        "type",
        "primary_boardid",
        "engine",
        "market",
    ]
    text_values = [safe_lower(row.get(column)) for column in text_fields]
    text_blob = " | ".join(value for value in text_values if value)

    secid_lower = safe_lower(row.get("secid"))
    type_lower = safe_lower(row.get("type"))
    group_lower = safe_lower(row.get("group"))
    name_lower = safe_lower(row.get("name"))
    shortname_lower = safe_lower(row.get("shortname"))

    include_text = any(hint in text_blob for hint in TEXT_INCLUDE_HINTS)
    option_like = any(hint in text_blob for hint in OPTION_HINTS)
    if not option_like and type_lower in OPTION_EXACT_HINTS:
        option_like = True
    if not option_like and group_lower in OPTION_EXACT_HINTS:
        option_like = True
    if not option_like and "opt" in secid_lower:
        option_like = True

    exclude_by_type = any(hint in type_lower for hint in EXCLUDE_HINTS)
    exclude_by_group = any(hint in group_lower for hint in EXCLUDE_HINTS)

    keep = include_text and option_like and not (exclude_by_type or exclude_by_group)

    reasons: list[str] = []
    if include_text:
        reasons.append("matched_text")
    if option_like:
        reasons.append("option_like")
    if exclude_by_type:
        reasons.append("exclude_type")
    if exclude_by_group:
        reasons.append("exclude_group")

    return keep, ",".join(reasons)


def filter_candidates(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if metadata_df.empty:
        return metadata_df.copy()

    working = metadata_df.copy()
    keep_flags: list[bool] = []
    reasons: list[str] = []

    for _, row in working.iterrows():
        keep, reason = candidate_reason(row)
        keep_flags.append(keep)
        reasons.append(reason)

    working["_keep_candidate"] = keep_flags
    working["_candidate_reason"] = reasons
    filtered = working.loc[working["_keep_candidate"]].copy()

    if "secid" in filtered.columns:
        filtered = filtered.dropna(subset=["secid"])
        filtered["secid"] = filtered["secid"].astype(str)
        filtered = filtered.drop_duplicates(subset=["secid"]).reset_index(drop=True)

    return filtered


def fetch_history_for_secid(
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
    history_columns: list[str] = []

    while True:
        payload = fetch_json(
            session,
            url,
            params,
            logger=logger,
            context=f"history:{secid}:start={params['start']}",
        )
        if payload is None:
            return pd.DataFrame(columns=history_columns or EXPECTED_HISTORY_COLUMNS), "request failed"

        frame = table_to_frame(payload, "history")
        if list(frame.columns) and not history_columns:
            history_columns = frame.columns.tolist()

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
        return pd.DataFrame(columns=history_columns or EXPECTED_HISTORY_COLUMNS), None

    combined = pd.concat(frames, ignore_index=True)
    return combined, None


def fetch_imoex_candles(
    session: requests.Session,
    logger: logging.Logger,
    diagnostics: list[str],
) -> pd.DataFrame:
    params = {
        "from": DATE_FROM,
        "till": DATE_TILL,
        "interval": 24,
        "iss.meta": "off",
        "limit": 1000,
    }
    payload = fetch_json(
        session,
        CANDLES_URL,
        params,
        logger=logger,
        context="imoex_candles",
    )
    if payload is None:
        diagnostics.append("IMOEX candles request failed")
        return pd.DataFrame()

    return table_to_frame(payload, "candles")


def format_date_range(df: pd.DataFrame, date_column: str) -> str:
    if df.empty or date_column not in df.columns:
        return "no rows"

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return "no valid dates"

    return f"{dates.min().date()} to {dates.max().date()}"


def format_rows_by_year(df: pd.DataFrame, date_column: str) -> str:
    if df.empty or date_column not in df.columns:
        return "none"

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return "none"

    counts = dates.dt.year.value_counts().sort_index()
    return ", ".join(f"{int(year)}: {int(count)}" for year, count in counts.items())


def format_top_secids(history_df: pd.DataFrame) -> str:
    if history_df.empty or "SECID" not in history_df.columns:
        return "none"

    counts = history_df["SECID"].astype(str).value_counts().head(20)
    return ", ".join(f"{secid}: {int(count)}" for secid, count in counts.items())


def save_frame(df: pd.DataFrame, parquet_path: Path, csv_path: Path | None = None) -> None:
    frame_to_save = df
    if frame_to_save.empty and not list(frame_to_save.columns):
        frame_to_save = pd.DataFrame({"note": []})

    frame_to_save.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        frame_to_save.to_csv(csv_path, index=False)


def build_report(
    discovery_urls: list[str],
    metadata_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    candidate_secids: list[str],
    history_df: pd.DataFrame,
    non_empty_history_count: int,
    candles_df: pd.DataFrame,
    failed_secids: list[str],
    diagnostics: list[str],
) -> str:
    lines = [
        "MOEX options 2y MVP report",
        "",
        "tested discovery URLs",
        *[f"- {url}" for url in discovery_urls],
        "",
        f"metadata columns: {', '.join(metadata_df.columns.tolist()) or 'none'}",
        f"number of raw metadata rows: {len(metadata_df)}",
        f"number of filtered candidates: {len(candidates_df)}",
        f"first 50 candidate SECIDs: {', '.join(candidate_secids) or 'none'}",
        f"number of SECIDs with non-empty history: {non_empty_history_count}",
        f"history columns: {', '.join(history_df.columns.tolist()) or 'none'}",
        f"total history rows: {len(history_df)}",
        f"date range: {format_date_range(history_df, 'TRADEDATE')}",
        f"rows by year: {format_rows_by_year(history_df, 'TRADEDATE')}",
        f"rows by SECID top 20: {format_top_secids(history_df)}",
        f"IMOEX candles row count: {len(candles_df)}",
        f"IMOEX candles date range: {format_date_range(candles_df, 'begin')}",
        "",
        "failed SECIDs with error messages",
    ]

    if failed_secids:
        lines.extend(f"- {item}" for item in failed_secids)
    else:
        lines.append("- none")

    if diagnostics:
        lines.extend(["", "diagnostics"])
        lines.extend(f"- {item}" for item in diagnostics)

    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    diagnostics: list[str] = []
    failed_secids: list[str] = []

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "download_moex_options_2y.py (requests/pandas research script)"}
    )

    logger.info("Starting narrow MOEX discovery using q=IMOEX, MIX, MX")
    metadata_df, discovery_urls = fetch_discovery_rows(session, logger, diagnostics)
    if metadata_df.empty:
        logger.warning("No metadata rows were discovered.")
        diagnostics.append("raw discovery returned zero rows")
    save_frame(metadata_df, RAW_METADATA_PATH)

    candidates_df = filter_candidates(metadata_df)
    if candidates_df.empty:
        logger.warning("No candidates matched the defensive filter.")
        diagnostics.append("candidate filter returned zero rows")
    save_frame(candidates_df, CANDIDATES_PATH)

    candidate_secids = []
    if "secid" in candidates_df.columns:
        candidate_secids = candidates_df["secid"].astype(str).head(MAX_CANDIDATE_SECIDS).tolist()
    if len(candidates_df) > MAX_CANDIDATE_SECIDS:
        diagnostics.append(
            f"candidate SECIDs truncated from {len(candidates_df)} to {MAX_CANDIDATE_SECIDS}"
        )

    history_frames: list[pd.DataFrame] = []
    non_empty_history_count = 0

    for secid in candidate_secids:
        history_df, error_message = fetch_history_for_secid(session, secid, logger)
        if error_message:
            failed_secids.append(f"{secid}: {error_message}")
        elif history_df.empty:
            diagnostics.append(f"{secid}: history returned zero rows")
        else:
            non_empty_history_count += 1
            history_frames.append(history_df)
        time.sleep(SLEEP_SECONDS)

    if history_frames:
        combined_history_df = pd.concat(history_frames, ignore_index=True)
        if "TRADEDATE" in combined_history_df.columns:
            combined_history_df = combined_history_df.sort_values(
                ["SECID", "TRADEDATE"], kind="stable"
            ).reset_index(drop=True)
    else:
        combined_history_df = pd.DataFrame(columns=EXPECTED_HISTORY_COLUMNS)
        diagnostics.append("no option history rows were found for the candidate SECIDs")
        logger.warning("No option history rows were found. Diagnostics will be saved to the report.")

    save_frame(combined_history_df, HISTORY_PARQUET_PATH, HISTORY_CSV_PATH)

    candles_df = fetch_imoex_candles(session, logger, diagnostics)
    if not candles_df.empty and "begin" in candles_df.columns:
        candles_df = candles_df.sort_values("begin", kind="stable").reset_index(drop=True)
    save_frame(candles_df, CANDLES_PARQUET_PATH, CANDLES_CSV_PATH)

    report_text = build_report(
        discovery_urls=discovery_urls,
        metadata_df=metadata_df,
        candidates_df=candidates_df,
        candidate_secids=candidate_secids,
        history_df=combined_history_df,
        non_empty_history_count=non_empty_history_count,
        candles_df=candles_df,
        failed_secids=failed_secids,
        diagnostics=diagnostics,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    logger.info("Saved raw metadata to %s", RAW_METADATA_PATH)
    logger.info("Saved filtered candidates to %s", CANDIDATES_PATH)
    logger.info("Saved option history to %s", HISTORY_PARQUET_PATH)
    logger.info("Saved IMOEX candles to %s", CANDLES_PARQUET_PATH)
    logger.info("Saved report to %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
