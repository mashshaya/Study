#!/usr/bin/env python3
"""Minimal sanity-check for MOEX historical option price availability."""

from pathlib import Path
import json
import logging
import re
import time

import pandas as pd
import requests


CANDIDATES_PATH = Path("data/raw/moex_options_2y_candidates.parquet")
REPORT_PATH = Path("data/raw/moex_option_history_source_sanity_report.txt")
SEARCH_URL = "https://iss.moex.com/iss/securities.json"
HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/futures/markets/options/securities/{secid}.json"
)

TIMEOUT = 20
RETRIES = 3
SLEEP_SECONDS = 0.2
PAGE_LIMIT = 100
MAX_BASE_SECIDS = 10
TARGET_RANGE = ("2023-01-01", "2024-12-31")
CONTROL_RANGE = ("2024-01-01", "2026-05-20")
TARGET_YEAR_VARIANTS = ["4", "3"]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("sanity_check_moex_option_history_source")
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
    date_from: str,
    date_till: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None]:
    url = HISTORY_URL.format(secid=secid)
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
        payload, error = fetch_json(
            session,
            url,
            params,
            logger=logger,
            context=f"history:{secid}:{date_from}:{date_till}:start={params['start']}",
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


def fetch_search_preview(
    session: requests.Session,
    query: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str | None]:
    params = {"iss.meta": "off", "q": query, "limit": 5}
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


def make_historical_variant(secid: str, target_year_digit: str) -> str | None:
    match = re.search(r"(\d)([A-Z]?)$", secid)
    if not match:
        return None
    start, end = match.span(1)
    return secid[:start] + target_year_digit + secid[end:]


def summarize_history(df: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": int(len(df)),
        "first_tradedate": None,
        "last_tradedate": None,
        "nonzero_settleprice": 0,
        "nonzero_volume": 0,
    }
    if df.empty:
        return summary

    working = df.copy()
    if "TRADEDATE" in working.columns:
        working["TRADEDATE"] = pd.to_datetime(working["TRADEDATE"], errors="coerce")
        dates = working["TRADEDATE"].dropna()
        if not dates.empty:
            summary["first_tradedate"] = str(dates.min().date())
            summary["last_tradedate"] = str(dates.max().date())

    if "SETTLEPRICE" in working.columns:
        settle = pd.to_numeric(working["SETTLEPRICE"], errors="coerce").fillna(0)
        summary["nonzero_settleprice"] = int((settle != 0).sum())

    if "VOLUME" in working.columns:
        volume = pd.to_numeric(working["VOLUME"], errors="coerce").fillna(0)
        summary["nonzero_volume"] = int((volume != 0).sum())

    return summary


def main() -> int:
    logger = setup_logger()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(CANDIDATES_PATH)
    base_secids = candidates["secid"].dropna().astype(str).head(MAX_BASE_SECIDS).tolist()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "sanity_check_moex_option_history_source.py "
                "(requests/pandas availability check)"
            )
        }
    )

    lines = [
        "MOEX option history source sanity-check",
        "",
        f"candidate source: {CANDIDATES_PATH}",
        f"base SECIDs tested: {len(base_secids)}",
        f"target range: {TARGET_RANGE[0]} -> {TARGET_RANGE[1]}",
        f"control range: {CONTROL_RANGE[0]} -> {CONTROL_RANGE[1]}",
        "",
        "Logic:",
        "- Test current SECIDs on target and control ranges.",
        "- Generate simple historical SECID variants by replacing the final year digit with 4 and 3.",
        "- For each variant, check exact search visibility and target-range history rows.",
        "- If any variant returns target-range rows, the source is technically usable.",
        "",
    ]

    current_with_control_rows = 0
    target_hits = 0
    search_hits = 0

    for secid in base_secids:
        lines.append(f"BASE SECID: {secid}")

        current_target_df, current_target_error = fetch_history(
            session,
            secid,
            TARGET_RANGE[0],
            TARGET_RANGE[1],
            logger,
        )
        current_control_df, current_control_error = fetch_history(
            session,
            secid,
            CONTROL_RANGE[0],
            CONTROL_RANGE[1],
            logger,
        )
        current_target_summary = summarize_history(current_target_df)
        current_control_summary = summarize_history(current_control_df)

        if current_control_summary["rows"] > 0:
            current_with_control_rows += 1

        lines.append("  current SECID")
        lines.append(
            f"    target rows={current_target_summary['rows']} "
            f"first={current_target_summary['first_tradedate']} "
            f"last={current_target_summary['last_tradedate']}"
        )
        lines.append(
            f"    control rows={current_control_summary['rows']} "
            f"first={current_control_summary['first_tradedate']} "
            f"last={current_control_summary['last_tradedate']}"
        )
        if current_target_error:
            lines.append(f"    target error: {current_target_error}")
        if current_control_error:
            lines.append(f"    control error: {current_control_error}")

        for year_digit in TARGET_YEAR_VARIANTS:
            variant = make_historical_variant(secid, year_digit)
            if not variant:
                lines.append(f"  variant year={year_digit}: could not generate")
                continue

            search_df, search_error = fetch_search_preview(session, variant, logger)
            history_df, history_error = fetch_history(
                session,
                variant,
                TARGET_RANGE[0],
                TARGET_RANGE[1],
                logger,
            )
            summary = summarize_history(history_df)

            if not search_df.empty:
                search_hits += 1
            if summary["rows"] > 0:
                target_hits += 1

            lines.append(f"  variant {variant}")
            lines.append(f"    search rows={len(search_df)}")
            if search_error:
                lines.append(f"    search error: {search_error}")
            lines.append(
                f"    target rows={summary['rows']} "
                f"first={summary['first_tradedate']} "
                f"last={summary['last_tradedate']} "
                f"nonzero_settle={summary['nonzero_settleprice']} "
                f"nonzero_volume={summary['nonzero_volume']}"
            )
            if history_error:
                lines.append(f"    history error: {history_error}")

            if not search_df.empty:
                preview_cols = [col for col in ["secid", "shortname", "name", "type", "group"] if col in search_df.columns]
                preview = search_df[preview_cols].head(3).to_dict(orient="records")
                for row in preview:
                    lines.append("    search preview: " + json.dumps(row, ensure_ascii=False))
            time.sleep(SLEEP_SECONDS)

        lines.append("")

    lines.extend(
        [
            "Overall result",
            f"- current SECIDs with control-range history: {current_with_control_rows} / {len(base_secids)}",
            f"- historical variants visible in exact search: {search_hits}",
            f"- historical variants with target-range history rows: {target_hits}",
        ]
    )

    if target_hits > 0:
        lines.append(
            "- Conclusion: source is technically usable for historical prices; discovery must be improved."
        )
    elif current_with_control_rows > 0:
        lines.append(
            "- Conclusion: endpoint works for current contracts, but this heuristic did not prove 2023-2024 history availability."
        )
        lines.append(
            "- Interpretation: likely discovery/identifier problem remains unresolved; historical availability is still unproven."
        )
    else:
        lines.append(
            "- Conclusion: even control checks failed, so the source may be unsuitable or the endpoint assumptions are wrong."
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report saved to %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
