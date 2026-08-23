#!/usr/bin/env python3
"""Try several strategies to find at least one historical MOEX option SECID."""

from pathlib import Path
import json
import logging
import re
import time

import pandas as pd
import requests


CANDIDATES_PATH = Path("data/raw/moex_options_2y_candidates.parquet")
REPORT_PATH = Path("data/raw/find_historical_moex_option_secids_report.txt")
SEARCH_URL = "https://iss.moex.com/iss/securities.json"
HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/futures/markets/options/securities/{secid}.json"
)

TIMEOUT = 20
RETRIES = 3
SLEEP_SECONDS = 0.2
SEARCH_LIMIT = 100
HISTORY_LIMIT = 100
BASE_ROWS = 12
MAX_DISCOVERED_TO_PROBE = 150
TARGET_FROM = "2023-01-01"
TARGET_TILL = "2024-12-31"
YEAR_SUFFIXES = ["5", "4", "3"]
SHORTNAME_YY = ["25", "24", "23"]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("find_historical_moex_option_secids")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    return logger


def build_url(url: str, params: dict[str, object]) -> str:
    request = requests.PreparedRequest()
    request.prepare_url(url, params)
    return request.url


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    *,
    logger: logging.Logger,
    context: str,
) -> tuple[dict | None, str | None]:
    full_url = build_url(url, params)
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            logger.info("GET %s attempt %s/%s: %s", context, attempt, RETRIES, full_url)
            response = session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json(), None
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("%s failed on attempt %s/%s: %s", context, attempt, RETRIES, exc)
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


def fetch_history(
    session: requests.Session,
    secid: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None]:
    url = HISTORY_URL.format(secid=secid)
    params = {
        "from": TARGET_FROM,
        "till": TARGET_TILL,
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

    return pd.concat(frames, ignore_index=True), None


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_lower(value: object) -> str:
    return safe_text(value).lower()


def is_option_like_row(row: pd.Series) -> bool:
    blob = " | ".join(
        safe_lower(row.get(column))
        for column in ["secid", "shortname", "name", "type", "group"]
    )
    include = any(token in blob for token in ["imoex", "mix", "mx", "индекс мосбиржи"])
    option_like = any(token in blob for token in ["option", "опцион", "call", "put", "futures_options"])
    return include and option_like


def generate_secid_variants(secid: str) -> list[str]:
    variants = [secid]
    match = re.search(r"(\d)([A-Z]?)$", secid)
    if not match:
        return variants
    start, end = match.span(1)
    for year_digit in YEAR_SUFFIXES:
        variants.append(secid[:start] + year_digit + secid[end:])
    return list(dict.fromkeys(variants))


def generate_shortname_variants(shortname: str) -> list[str]:
    value = safe_text(shortname)
    if not value:
        return []
    variants = [value]
    date_match = re.search(r"(\d{4})(\d{2})", value)
    if date_match:
        start, end = date_match.span()
        ddmm = date_match.group(1)
        for yy in SHORTNAME_YY:
            variants.append(value[:start] + ddmm + yy + value[end:])
    return list(dict.fromkeys(variants))


def generate_queries(row: pd.Series) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []

    secid = safe_text(row.get("secid"))
    shortname = safe_text(row.get("shortname"))
    name = safe_text(row.get("name"))

    for variant in generate_secid_variants(secid):
        queries.append(("secid_variant", variant))

    for variant in generate_shortname_variants(shortname):
        queries.append(("shortname_variant", variant))

    if shortname:
        prefix_match = re.match(r"([A-Z]+)", shortname)
        if prefix_match:
            queries.append(("shortname_prefix", prefix_match.group(1)))

    if "IMOEX" in shortname:
        queries.append(("fixed_query", "IMOEXP"))
        queries.append(("fixed_query", "IMOEX option"))

    if name:
        parts = [part.strip() for part in re.split(r"с исп\\.|на IMOEX", name) if part.strip()]
        if parts:
            queries.append(("name_fragment", parts[0]))

    deduped = []
    seen = set()
    for strategy, query in queries:
        key = (strategy, query)
        if query and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def search_query(
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


def summarize_history(df: pd.DataFrame) -> dict[str, object]:
    summary = {"rows": int(len(df)), "first": None, "last": None}
    if df.empty:
        return summary
    working = df.copy()
    if "TRADEDATE" in working.columns:
        working["TRADEDATE"] = pd.to_datetime(working["TRADEDATE"], errors="coerce")
        dates = working["TRADEDATE"].dropna()
        if not dates.empty:
            summary["first"] = str(dates.min().date())
            summary["last"] = str(dates.max().date())
    return summary


def main() -> int:
    logger = setup_logger()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_parquet(CANDIDATES_PATH).head(BASE_ROWS).copy()
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "find_historical_moex_option_secids.py (requests/pandas probe)"}
    )

    search_results_frames: list[pd.DataFrame] = []
    attempted_queries: list[tuple[str, str, int, str | None]] = []

    for _, row in base_df.iterrows():
        for strategy, query in generate_queries(row):
            result_df, error = search_query(session, query, logger)
            attempted_queries.append((strategy, query, len(result_df), error))
            if not result_df.empty:
                result_df = result_df.copy()
                result_df["_strategy"] = strategy
                result_df["_query"] = query
                search_results_frames.append(result_df)
            time.sleep(SLEEP_SECONDS)

    if search_results_frames:
        search_df = pd.concat(search_results_frames, ignore_index=True)
        keep_cols = [col for col in ["secid", "shortname", "name", "type", "group"] if col in search_df.columns]
        if keep_cols:
            search_df = search_df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
        filtered_search_df = search_df.loc[search_df.apply(is_option_like_row, axis=1)].copy()
    else:
        search_df = pd.DataFrame()
        filtered_search_df = pd.DataFrame()

    direct_probe_secids = []
    for secid in base_df["secid"].astype(str).tolist():
        direct_probe_secids.extend(generate_secid_variants(secid))
    direct_probe_secids = list(dict.fromkeys(direct_probe_secids))

    discovered_secids = []
    if "secid" in filtered_search_df.columns:
        discovered_secids = filtered_search_df["secid"].dropna().astype(str).tolist()

    probe_secids = list(dict.fromkeys(direct_probe_secids + discovered_secids))
    probe_secids = probe_secids[:MAX_DISCOVERED_TO_PROBE]

    history_hits: list[tuple[str, int, str | None, str | None, str | None]] = []
    empty_hits: list[str] = []
    errors: list[tuple[str, str]] = []

    for secid in probe_secids:
        history_df, error = fetch_history(session, secid, logger)
        if error:
            errors.append((secid, error))
        else:
            summary = summarize_history(history_df)
            if summary["rows"] > 0:
                history_hits.append((secid, summary["rows"], summary["first"], summary["last"], None))
            else:
                empty_hits.append(secid)
        time.sleep(SLEEP_SECONDS)

    lines = [
        "Find historical MOEX option SECIDs report",
        "",
        f"candidate source: {CANDIDATES_PATH}",
        f"base rows used: {len(base_df)}",
        f"target range: {TARGET_FROM} -> {TARGET_TILL}",
        "",
        "Attempted search queries",
    ]
    for strategy, query, rows, error in attempted_queries:
        line = f"- {strategy}: {query!r} -> rows={rows}"
        if error:
            line += f" error={error}"
        lines.append(line)

    lines.extend(
        [
            "",
            f"raw search result rows: {len(search_df)}",
            f"filtered option-like search rows: {len(filtered_search_df)}",
            f"SECIDs probed on history endpoint: {len(probe_secids)}",
            "",
            "History hits on target range",
        ]
    )

    if history_hits:
        for secid, rows, first, last, _ in history_hits:
            lines.append(f"- {secid}: rows={rows}, first={first}, last={last}")
    else:
        lines.append("- none")

    lines.extend(["", "History errors"])
    if errors:
        for secid, error in errors:
            lines.append(f"- {secid}: {error}")
    else:
        lines.append("- none")

    lines.extend(["", "Interpretation"])
    if history_hits:
        lines.append(
            "- Success: at least one SECID returned rows for 2023-2024, so MOEX ISS is technically usable."
        )
        lines.append(
            "- Next step: use the hit SECIDs and neighboring search patterns to expand discovery."
        )
    else:
        lines.append(
            "- No target-range hits were found with these heuristics."
        )
        lines.append(
            "- This does not prove the source is impossible, but it means the current identifier-search strategies were insufficient."
        )
        lines.append(
            "- If this remains true after broader manual lookup, another source may be needed."
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report saved to %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
