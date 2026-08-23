#!/usr/bin/env python3
"""Download an MVP historical dataset for MOEX IMOEX-related options.

This script discovers futures and options securities through the MOEX ISS API,
filters candidate options related to the MOEX Russia Index / IMOEX, downloads
daily historical data for 2023-01-01..2024-12-31, and writes:

- data/raw/moex_imoex_options_metadata_mvp.parquet
- data/raw/moex_imoex_options_daily_mvp.parquet
- data/raw/moex_imoex_options_daily_mvp.csv
- data/raw/moex_imoex_options_report_mvp.txt

Dependencies:
- Python 3.12
- requests
- pandas
- pyarrow
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE_URL = "https://iss.moex.com/iss"
DATE_FROM = "2023-01-01"
DATE_TILL = "2024-12-31"

OUTPUT_DIR = Path("data/raw")
METADATA_PATH = OUTPUT_DIR / "moex_imoex_options_metadata_mvp.parquet"
DAILY_PARQUET_PATH = OUTPUT_DIR / "moex_imoex_options_daily_mvp.parquet"
DAILY_CSV_PATH = OUTPUT_DIR / "moex_imoex_options_daily_mvp.csv"
REPORT_PATH = OUTPUT_DIR / "moex_imoex_options_report_mvp.txt"

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5
REQUEST_SLEEP_SECONDS = 0.20
PAGE_SIZE = 100
DISCOVERY_MAX_PAGES = 20
CANDIDATE_LIMIT = 200

SECURITY_SEARCH_URL = f"{BASE_URL}/securities.json"
OPTIONS_DISCOVERY_URL = f"{BASE_URL}/engines/futures/markets/options/securities.json"
FUTURES_DISCOVERY_URL = f"{BASE_URL}/engines/futures/markets/forts/securities.json"
HISTORY_URL_TEMPLATE = (
    f"{BASE_URL}/history/engines/futures/markets/options/securities/{{secid}}.json"
)
HISTORY_BOARD_URL_TEMPLATE = (
    f"{BASE_URL}/history/engines/futures/markets/options/boards/{{boardid}}"
    "/securities/{secid}.json"
)
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

STRONG_TEXT_HINTS = [
    "imoex",
    "moex index",
    "russia index",
    "индекс мосбир",
    "московской бирж",
]
CODE_HINTS = ["IMOEX", "MIX", "MX", "MXI"]
DISCOVERY_QUERIES = ["IMOEX", "MIX", "MX"]


def log(message: str) -> None:
    print(message, file=sys.stderr)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_url(url: str, params: dict[str, object] | None = None) -> str:
    request = requests.PreparedRequest()
    request.prepare_url(url, params or {})
    return request.url


def safe_upper(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def safe_lower(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    *,
    context: str,
    diagnostics: list[str],
) -> dict | None:
    full_url = build_url(url, params)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"[info] GET {context} attempt {attempt}/{MAX_RETRIES}: {full_url}")
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            diagnostics.append(
                f"{context} attempt {attempt}/{MAX_RETRIES} failed for {full_url}: {exc}"
            )
            log(
                f"[warn] {context} attempt {attempt}/{MAX_RETRIES} failed: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    diagnostics.append(f"{context} exhausted retries for {full_url}: {last_error}")
    return None


def table_to_frame(payload: dict | None, table_name: str) -> pd.DataFrame:
    if not payload or table_name not in payload:
        return pd.DataFrame()

    block = payload.get(table_name) or {}
    columns = block.get("columns") or []
    data = block.get("data") or []
    return pd.DataFrame(data, columns=columns)


def cursor_info(payload: dict | None, table_name: str) -> tuple[int | None, int | None]:
    if not payload:
        return None, None

    block = payload.get(f"{table_name}.cursor")
    if not isinstance(block, dict):
        return None, None

    columns = block.get("columns") or []
    data = block.get("data") or []
    if not columns or not data:
        return None, None

    row = dict(zip(columns, data[0]))
    total = row.get("TOTAL")
    page_size = row.get("PAGESIZE")

    try:
        total_value = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_value = None

    try:
        page_size_value = int(page_size) if page_size is not None else None
    except (TypeError, ValueError):
        page_size_value = None

    return total_value, page_size_value


def fetch_paginated_table(
    session: requests.Session,
    url: str,
    *,
    table_name: str,
    context: str,
    params: dict[str, object],
    diagnostics: list[str],
    max_pages: int | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start = 0
    page_size = PAGE_SIZE
    saw_response = False
    last_frame = pd.DataFrame()
    pages_fetched = 0

    while True:
        if max_pages is not None and pages_fetched >= max_pages:
            diagnostics.append(
                f"{context} stopped after reaching max_pages={max_pages}"
            )
            log(f"[warn] {context} reached max_pages={max_pages}, stopping pagination.")
            break

        page_params = dict(params)
        page_params["start"] = start

        payload = fetch_json(
            session,
            url,
            page_params,
            context=context,
            diagnostics=diagnostics,
        )
        if payload is None:
            break

        saw_response = True
        pages_fetched += 1
        frame = table_to_frame(payload, table_name)
        last_frame = frame
        total, page_size_from_cursor = cursor_info(payload, table_name)
        log(
            f"[info] {context} page={pages_fetched} start={start} "
            f"rows={len(frame)} total={total if total is not None else 'unknown'}"
        )

        if page_size_from_cursor:
            page_size = page_size_from_cursor

        if frame.empty:
            break

        frames.append(frame)
        row_count = len(frame)

        if total is not None and start + row_count >= total:
            break

        if row_count < page_size:
            break

        start += row_count
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not frames:
        if saw_response:
            return last_frame
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["_source_context"] = context
    return combined


def discover_security_searches(
    session: requests.Session,
    diagnostics: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for query in DISCOVERY_QUERIES:
        params = {
            "iss.meta": "off",
            "iss.only": "securities,securities.cursor",
            "q": query,
            "limit": PAGE_SIZE,
        }
        frame = fetch_paginated_table(
            session,
            SECURITY_SEARCH_URL,
            table_name="securities",
            context=f"discover:security-search:{query}",
            params=params,
            diagnostics=diagnostics,
            max_pages=DISCOVERY_MAX_PAGES,
        )
        log(f"[info] Query {query!r} returned {len(frame)} rows.")
        diagnostics.append(f"security search query {query}: {len(frame)} rows")
        if not frame.empty:
            frame = frame.copy()
            frame["_search_query"] = query
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    dedupe_columns = [column for column in ["secid", "primary_boardid", "name"] if column in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    else:
        combined = combined.drop_duplicates().reset_index(drop=True)
    return combined


def contains_strong_text_hint(values: Iterable[object]) -> list[str]:
    text = " | ".join(safe_lower(value) for value in values if safe_lower(value))
    reasons = [f"text:{hint}" for hint in STRONG_TEXT_HINTS if hint in text]
    return reasons


def contains_code_hint(values: Iterable[object]) -> list[str]:
    reasons: list[str] = []

    for value in values:
        upper_value = safe_upper(value)
        if not upper_value:
            continue

        if upper_value.startswith("IMOEX"):
            reasons.append("code:IMOEX")
        if upper_value.startswith("MIX"):
            reasons.append("code:MIX")
        if upper_value.startswith("MXI"):
            reasons.append("code:MXI")
        if upper_value == "MX" or upper_value.startswith("MX-") or upper_value.startswith("MX"):
            reasons.append("code:MX")

    return reasons


def normalize_reason_list(reasons: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for reason in reasons:
        if reason and reason not in seen:
            seen.add(reason)
            result.append(reason)

    return result


def score_futures_candidates(futures_df: pd.DataFrame) -> pd.DataFrame:
    if futures_df.empty:
        futures_df = pd.DataFrame(
            columns=["SECID", "BOARDID", "SHORTNAME", "SECNAME", "LATNAME", "ASSETCODE"]
        )

    working = futures_df.copy()
    reasons_column: list[list[str]] = []

    for _, row in working.iterrows():
        text_values = [
            row.get("SECID"),
            row.get("SHORTNAME"),
            row.get("SECNAME"),
            row.get("LATNAME"),
            row.get("ASSETCODE"),
        ]
        code_values = [
            row.get("SECID"),
            row.get("SHORTNAME"),
            row.get("LATNAME"),
            row.get("ASSETCODE"),
        ]
        reasons = contains_strong_text_hint(text_values) + contains_code_hint(code_values)
        reasons_column.append(normalize_reason_list(reasons))

    working["_future_candidate_reasons"] = reasons_column
    working["_is_relevant_future"] = working["_future_candidate_reasons"].map(bool)
    working["_future_candidate_reason"] = working["_future_candidate_reasons"].map(", ".join)
    return working


def score_option_candidates(
    options_df: pd.DataFrame,
    relevant_futures_df: pd.DataFrame,
) -> pd.DataFrame:
    if options_df.empty:
        options_df = pd.DataFrame(
            columns=[
                "SECID",
                "BOARDID",
                "SHORTNAME",
                "SECNAME",
                "LATNAME",
                "ASSETCODE",
                "OPTIONTYPE",
                "UNDERLYINGASSET",
                "UNDERLYINGTYPE",
            ]
        )

    working = options_df.copy()
    relevant_future_secids = {
        safe_upper(value)
        for value in relevant_futures_df.get("SECID", pd.Series(dtype=object))
        if safe_upper(value)
    }
    relevant_future_assetcodes = {
        safe_upper(value)
        for value in relevant_futures_df.get("ASSETCODE", pd.Series(dtype=object))
        if safe_upper(value)
    }

    candidate_reason_lists: list[list[str]] = []
    matched_future_secids: list[str] = []
    matched_future_assetcodes: list[str] = []

    for _, row in working.iterrows():
        reasons: list[str] = []
        matches_secids: list[str] = []
        matches_assetcodes: list[str] = []

        underlying_type = safe_upper(row.get("UNDERLYINGTYPE"))
        underlying_asset = safe_upper(row.get("UNDERLYINGASSET"))
        assetcode = safe_upper(row.get("ASSETCODE"))

        if underlying_type != "F":
            candidate_reason_lists.append([])
            matched_future_secids.append("")
            matched_future_assetcodes.append("")
            continue

        text_values = [
            row.get("SECID"),
            row.get("SHORTNAME"),
            row.get("SECNAME"),
            row.get("LATNAME"),
            row.get("ASSETCODE"),
            row.get("UNDERLYINGASSET"),
        ]
        code_values = [
            row.get("SECID"),
            row.get("ASSETCODE"),
            row.get("UNDERLYINGASSET"),
        ]
        reasons.extend(contains_strong_text_hint(text_values))
        reasons.extend(contains_code_hint(code_values))

        if underlying_asset and underlying_asset in relevant_future_secids:
            matches_secids.append(underlying_asset)
            reasons.append("underlyingasset:matched_relevant_future_secid")

        if assetcode and assetcode in relevant_future_assetcodes:
            matches_assetcodes.append(assetcode)
            reasons.append("assetcode:matched_relevant_future_assetcode")

        reasons = normalize_reason_list(reasons)
        candidate_reason_lists.append(reasons)
        matched_future_secids.append(",".join(sorted(set(matches_secids))))
        matched_future_assetcodes.append(",".join(sorted(set(matches_assetcodes))))

    working["_candidate_reasons"] = candidate_reason_lists
    working["_candidate_reason"] = working["_candidate_reasons"].map(", ".join)
    working["_matched_future_secids"] = matched_future_secids
    working["_matched_future_assetcodes"] = matched_future_assetcodes
    working["_is_candidate"] = working["_candidate_reasons"].map(bool)
    return working


def apply_mvp_search_filter(
    metadata_df: pd.DataFrame,
    search_df: pd.DataFrame,
) -> pd.DataFrame:
    if metadata_df.empty:
        return metadata_df.copy()

    if search_df.empty:
        working = metadata_df.copy()
        working["_search_seed_match"] = True
        return working

    search_secids = {
        safe_upper(value)
        for column in ["secid"]
        if column in search_df.columns
        for value in search_df[column].dropna().tolist()
        if safe_upper(value)
    }
    search_names = {
        safe_lower(value)
        for column in ["shortname", "name", "secname"]
        if column in search_df.columns
        for value in search_df[column].dropna().tolist()
        if safe_lower(value)
    }

    matches: list[bool] = []

    for _, row in metadata_df.iterrows():
        upper_tokens = {
            safe_upper(row.get("SECID")),
            safe_upper(row.get("ASSETCODE")),
            safe_upper(row.get("UNDERLYINGASSET")),
        }
        text_blob = " | ".join(
            safe_lower(row.get(column))
            for column in ["SECID", "SHORTNAME", "SECNAME", "LATNAME", "ASSETCODE", "UNDERLYINGASSET"]
            if column in metadata_df.columns
        )

        matched = bool(upper_tokens & search_secids)
        if not matched:
            matched = any(term.lower() in text_blob for term in DISCOVERY_QUERIES)
        if not matched and search_names:
            matched = any(name in text_blob for name in search_names if len(name) >= 3)
        matches.append(matched)

    working = metadata_df.copy()
    working["_search_seed_match"] = matches
    return working


def fetch_security_history(
    session: requests.Session,
    secid: str,
    boardid: str,
    diagnostics: list[str],
) -> tuple[pd.DataFrame, str]:
    common_params = {
        "iss.meta": "off",
        "iss.only": "history,history.cursor",
        "from": DATE_FROM,
        "till": DATE_TILL,
    }

    candidates = [
        (
            HISTORY_URL_TEMPLATE.format(secid=secid),
            f"history:{secid}:generic",
            "generic",
        )
    ]
    if boardid:
        candidates.append(
            (
                HISTORY_BOARD_URL_TEMPLATE.format(boardid=boardid, secid=secid),
                f"history:{secid}:board:{boardid}",
                "board-specific",
            )
        )

    last_empty_frame = pd.DataFrame()

    for url, context, label in candidates:
        frame = fetch_paginated_table(
            session,
            url,
            table_name="history",
            context=context,
            params=common_params,
            diagnostics=diagnostics,
        )
        if not frame.empty:
            frame["_history_endpoint_label"] = label
            frame["_history_secid"] = secid
            return frame, label
        last_empty_frame = frame

    diagnostics.append(f"no history rows returned for SECID={secid}, BOARDID={boardid}")
    return last_empty_frame, "none"


def format_rows_by_year(history_df: pd.DataFrame) -> str:
    if history_df.empty or "TRADEDATE" not in history_df.columns:
        return "none"

    trade_dates = pd.to_datetime(history_df["TRADEDATE"], errors="coerce")
    year_counts = Counter(int(year) for year in trade_dates.dt.year.dropna().tolist())
    if not year_counts:
        return "none"

    return ", ".join(f"{year}: {count}" for year, count in sorted(year_counts.items()))


def format_date_range(history_df: pd.DataFrame) -> str:
    if history_df.empty or "TRADEDATE" not in history_df.columns:
        return "no rows"

    trade_dates = pd.to_datetime(history_df["TRADEDATE"], errors="coerce").dropna()
    if trade_dates.empty:
        return "no valid dates"

    return f"{trade_dates.min().date()} to {trade_dates.max().date()}"


def save_empty_parquet_like(df: pd.DataFrame, path: Path) -> None:
    if df.empty and not list(df.columns):
        df = pd.DataFrame({"note": []})
    df.to_parquet(path, index=False)


def build_report(
    *,
    tested_endpoints: list[str],
    discovered_options_count: int,
    candidate_count: int,
    metadata_df: pd.DataFrame,
    history_df: pd.DataFrame,
    candidate_secids: list[str],
    diagnostics: list[str],
) -> str:
    metadata_columns = ", ".join(metadata_df.columns.tolist()) or "none"
    history_columns = ", ".join(history_df.columns.tolist()) or "none"
    tested = "\n".join(f"- {endpoint}" for endpoint in tested_endpoints) or "- none"
    first_20_secids = ", ".join(candidate_secids[:20]) or "none"

    lines = [
        "MOEX IMOEX options MVP report",
        "",
        "tested endpoints",
        tested,
        "",
        f"number of discovered securities: {discovered_options_count}",
        f"number of candidate securities: {candidate_count}",
        f"metadata columns: {metadata_columns}",
        f"history columns: {history_columns}",
        f"total downloaded rows: {len(history_df)}",
        f"date range: {format_date_range(history_df)}",
        f"rows by year: {format_rows_by_year(history_df)}",
        f"first 20 candidate SECIDs: {first_20_secids}",
    ]

    if diagnostics:
        lines.extend(
            [
                "",
                "diagnostics",
                *[f"- {message}" for message in diagnostics],
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ensure_output_dir()

    diagnostics: list[str] = []
    tested_endpoints = [
        SECURITY_SEARCH_URL,
        OPTIONS_DISCOVERY_URL,
        FUTURES_DISCOVERY_URL,
        HISTORY_URL_TEMPLATE,
        HISTORY_BOARD_URL_TEMPLATE,
    ]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "download_moex_imoex_options_mvp.py "
                "(requests; contact: local-research-script)"
            )
        }
    )

    discovery_params = {
        "iss.meta": "off",
        "iss.only": "securities,securities.cursor",
        "limit": PAGE_SIZE,
    }

    log("[info] Discovering search seeds from /iss/securities.json...")
    search_df = discover_security_searches(session, diagnostics)

    log("[info] Discovering options securities from MOEX ISS...")
    options_df = fetch_paginated_table(
        session,
        OPTIONS_DISCOVERY_URL,
        table_name="securities",
        context="discover:options",
        params=discovery_params,
        diagnostics=diagnostics,
        max_pages=DISCOVERY_MAX_PAGES,
    )

    log("[info] Discovering futures securities from MOEX ISS...")
    futures_df = fetch_paginated_table(
        session,
        FUTURES_DISCOVERY_URL,
        table_name="securities",
        context="discover:futures",
        params=discovery_params,
        diagnostics=diagnostics,
        max_pages=DISCOVERY_MAX_PAGES,
    )

    if options_df.empty:
        diagnostics.append("options discovery returned no rows")
        log("[warn] Options discovery returned no rows.")

    if futures_df.empty:
        diagnostics.append("futures discovery returned no rows")
        log("[warn] Futures discovery returned no rows.")

    futures_scored_df = score_futures_candidates(futures_df)
    relevant_futures_df = futures_scored_df.loc[
        futures_scored_df["_is_relevant_future"]
    ].copy()

    metadata_df = score_option_candidates(options_df, relevant_futures_df)
    metadata_df = apply_mvp_search_filter(metadata_df, search_df)
    candidate_df = metadata_df.loc[
        metadata_df["_is_candidate"] & metadata_df["_search_seed_match"]
    ].copy()

    if len(candidate_df) > CANDIDATE_LIMIT:
        log(
            f"[warn] Candidate set has {len(candidate_df)} rows; "
            f"keeping the first {CANDIDATE_LIMIT} for MVP."
        )
        diagnostics.append(
            f"candidate set truncated from {len(candidate_df)} to {CANDIDATE_LIMIT} rows"
        )
        candidate_df = candidate_df.head(CANDIDATE_LIMIT).copy()

    if not relevant_futures_df.empty:
        diagnostics.append(
            "relevant futures assetcodes: "
            + ", ".join(sorted(relevant_futures_df["ASSETCODE"].dropna().astype(str).unique()))
        )
    else:
        diagnostics.append("no relevant futures matched the IMOEX/MIX/MX/MOEX Index filter")

    log(
        "[info] Discovery summary: "
        f"{len(search_df)} search-seed rows, "
        f"{len(options_df)} option securities, "
        f"{len(futures_df)} futures securities, "
        f"{len(candidate_df)} option candidates."
    )
    diagnostics.append(
        "discovery counts: "
        f"search_seed_rows={len(search_df)}, options={len(options_df)}, "
        f"futures={len(futures_df)}, candidates={len(candidate_df)}"
    )

    save_empty_parquet_like(metadata_df, METADATA_PATH)

    history_frames: list[pd.DataFrame] = []
    history_columns_reference: list[str] = []

    if candidate_df.empty:
        log("[warn] No candidate option securities matched the current filter.")
        diagnostics.append("no candidate option securities found after filtering")
    else:
        unique_candidates = (
            candidate_df[["SECID", "BOARDID"]]
            .dropna(subset=["SECID"])
            .drop_duplicates()
            .sort_values(["SECID", "BOARDID"], na_position="last")
        )

        for row in unique_candidates.itertuples(index=False):
            secid = str(row.SECID)
            boardid = "" if pd.isna(row.BOARDID) else str(row.BOARDID)

            log(f"[info] Downloading history for {secid}...")
            history_df, endpoint_label = fetch_security_history(
                session,
                secid=secid,
                boardid=boardid,
                diagnostics=diagnostics,
            )

            if list(history_df.columns) and not history_columns_reference:
                history_columns_reference = history_df.columns.tolist()

            if history_df.empty:
                diagnostics.append(
                    f"history returned zero rows for SECID={secid} via {endpoint_label}"
                )
                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            history_frames.append(history_df)
            time.sleep(REQUEST_SLEEP_SECONDS)

    if history_frames:
        combined_history_df = pd.concat(history_frames, ignore_index=True)
    else:
        combined_history_df = pd.DataFrame(
            columns=history_columns_reference or EXPECTED_HISTORY_COLUMNS
        )

    if not combined_history_df.empty:
        sort_columns = [column for column in ["SECID", "TRADEDATE", "BOARDID"] if column in combined_history_df.columns]
        if sort_columns:
            combined_history_df = combined_history_df.sort_values(sort_columns).reset_index(drop=True)

    save_empty_parquet_like(combined_history_df, DAILY_PARQUET_PATH)
    combined_history_df.to_csv(DAILY_CSV_PATH, index=False)

    candidate_secids = sorted(
        candidate_df["SECID"].dropna().astype(str).drop_duplicates().tolist()
    )
    report_text = build_report(
        tested_endpoints=tested_endpoints,
        discovered_options_count=len(options_df),
        candidate_count=len(candidate_df),
        metadata_df=metadata_df,
        history_df=combined_history_df,
        candidate_secids=candidate_secids,
        diagnostics=diagnostics,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    if combined_history_df.empty:
        log("[warn] No historical rows were downloaded. See the report for diagnostics:")
        log(f"       {REPORT_PATH}")
    else:
        log(
            "[info] Download completed: "
            f"{len(combined_history_df)} rows across "
            f"{combined_history_df['SECID'].nunique() if 'SECID' in combined_history_df.columns else 'n/a'} SECIDs."
        )
        log(f"[info] Report written to {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
