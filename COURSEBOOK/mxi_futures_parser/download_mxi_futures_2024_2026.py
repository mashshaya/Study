from pathlib import Path
import json
import time

import pandas as pd
import requests


BASE_URL = "https://iss.moex.com/iss"
SEARCH_URL = f"{BASE_URL}/securities.json"
HISTORY_URL_TEMPLATE = (
    f"{BASE_URL}/history/engines/futures/markets/forts/securities/{{secid}}.json"
)

DATE_FROM = "2024-01-01"
DATE_TILL = "2026-12-31"
TIMEOUT = 20
RETRIES = 3
LIMIT = 100
SLEEP_SECONDS = 0.2

OUTPUT_DIR = Path("/Users/maria/Desktop/Code/HSE/COURSEBOOK/data/mxi_futures")
METADATA_PATH = OUTPUT_DIR / "mxi_futures_metadata_2024_2026.parquet"
HISTORY_PARQUET_PATH = OUTPUT_DIR / "mxi_futures_daily_2024_2026.parquet"
HISTORY_CSV_PATH = OUTPUT_DIR / "mxi_futures_daily_2024_2026.csv"
REPORT_PATH = OUTPUT_DIR / "mxi_futures_report_2024_2026.txt"

SEARCH_QUERIES = ["MXI", "фьюч. контр. MXI", "MXI-"]


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
    return pd.DataFrame(block.get("data") or [], columns=block.get("columns") or [])


def cursor_to_dict(payload: dict | None, table_name: str) -> dict[str, object]:
    if not payload:
        return {}
    block = payload.get(f"{table_name}.cursor") or {}
    columns = block.get("columns") or []
    data = block.get("data") or []
    if not columns or not data:
        return {}
    return dict(zip(columns, data[0]))


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    context: str,
) -> tuple[dict | None, str | None]:
    full_url = build_url(url, params)
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            print(f"GET {context} attempt {attempt}/{RETRIES}: {full_url}")
            response = session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json(), None
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"  failed: {exc}")
            if attempt < RETRIES:
                time.sleep(SLEEP_SECONDS)

    return None, str(last_error)


def discover_mxi_futures(session: requests.Session) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    diagnostics = []

    for query in SEARCH_QUERIES:
        params = {"iss.meta": "off", "q": query, "limit": LIMIT}
        payload, error = fetch_json(session, SEARCH_URL, params, f"search:{query}")
        if payload is None:
            diagnostics.append(f"search query={query}: ERROR: {error}")
            continue

        frame = table_to_frame(payload, "securities")
        diagnostics.append(f"search query={query}: rows={len(frame)}")
        if frame.empty:
            continue
        frame["_query"] = query
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), diagnostics

    raw = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()

    secid_col = "secid" if "secid" in raw.columns else "SECID"
    shortname_col = "shortname" if "shortname" in raw.columns else None
    name_col = "name" if "name" in raw.columns else None
    type_col = "type" if "type" in raw.columns else None
    group_col = "group" if "group" in raw.columns else None

    def row_text(row: pd.Series) -> str:
        parts = [
            safe_text(row.get(secid_col)),
            safe_text(row.get(shortname_col)) if shortname_col else "",
            safe_text(row.get(name_col)) if name_col else "",
            safe_text(row.get(type_col)) if type_col else "",
            safe_text(row.get(group_col)) if group_col else "",
        ]
        return " ".join(part for part in parts if part).lower()

    text_blob = raw.apply(row_text, axis=1)

    looks_like_mxi = text_blob.str.contains("mxi", regex=False)
    looks_like_future = (
        text_blob.str.contains("futures", regex=False)
        | text_blob.str.contains("future", regex=False)
        | text_blob.str.contains("фьюч", regex=False)
        | text_blob.str.contains("forts", regex=False)
    )
    not_option = ~text_blob.str.contains("option|опцион|call|put|марж\\.|прем\\.", regex=True)
    not_index_spot = ~text_blob.str.contains("index|индекс", regex=True)

    filtered = raw.loc[looks_like_mxi & looks_like_future & not_option & not_index_spot].copy()
    if filtered.empty:
        diagnostics.append("filtered candidates: 0")
        return filtered, diagnostics

    filtered = filtered.drop_duplicates(subset=[secid_col]).reset_index(drop=True)
    diagnostics.append(f"filtered candidates: {len(filtered)}")
    diagnostics.append(
        "first candidate SECIDs: "
        + ", ".join(filtered[secid_col].astype(str).head(15).tolist())
    )
    return filtered, diagnostics


def fetch_full_history(
    session: requests.Session,
    secid: str,
) -> tuple[pd.DataFrame, str | None]:
    url = HISTORY_URL_TEMPLATE.format(secid=secid)
    params = {
        "from": DATE_FROM,
        "till": DATE_TILL,
        "iss.meta": "off",
        "start": 0,
        "limit": LIMIT,
    }
    frames = []
    history_columns: list[str] = []

    while True:
        payload, error = fetch_json(
            session,
            url,
            params,
            f"history:{secid}:start={params['start']}",
        )
        if payload is None:
            return pd.DataFrame(columns=history_columns), error

        frame = table_to_frame(payload, "history")
        if not history_columns and not frame.empty:
            history_columns = frame.columns.tolist()

        if frame.empty:
            break

        frames.append(frame)
        cursor = cursor_to_dict(payload, "history")
        total = cursor.get("TOTAL")
        page_size = cursor.get("PAGESIZE")
        next_start = int(params["start"]) + len(frame)

        if total is None or page_size is None:
            if len(frame) < LIMIT:
                break
        else:
            try:
                total_int = int(total)
                page_size_int = int(page_size)
            except (TypeError, ValueError):
                if len(frame) < LIMIT:
                    break
            else:
                if next_start >= total_int or len(frame) < page_size_int:
                    break

        params["start"] = next_start
        time.sleep(SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame(columns=history_columns), None

    return pd.concat(frames, ignore_index=True), None


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "mxi-futures-parser/1.0 (requests/pandas research script)"}
    )

    metadata, diagnostics = discover_mxi_futures(session)
    report_lines = [
        "MXI futures parser report",
        f"date range: {DATE_FROM} .. {DATE_TILL}",
        f"search queries: {', '.join(SEARCH_QUERIES)}",
        "",
        *diagnostics,
        "",
    ]

    if metadata.empty:
        REPORT_PATH.write_text("\n".join(report_lines))
        print("No MXI futures candidates found.")
        print(f"Saved report to {REPORT_PATH}")
        return 1

    metadata.to_parquet(METADATA_PATH, index=False)

    secid_col = "secid" if "secid" in metadata.columns else "SECID"
    history_frames = []
    failed = []

    for secid in metadata[secid_col].astype(str).tolist():
        frame, error = fetch_full_history(session, secid)
        if error:
            failed.append(f"{secid}: {error}")
            continue
        if frame.empty:
            print(f"No history rows for {secid}")
            continue
        frame["SECID_REQUESTED"] = secid
        history_frames.append(frame)

    if not history_frames:
        report_lines.extend(
            [
                "Downloaded history rows: 0",
                f"failed SECIDs: {len(failed)}",
                *failed,
            ]
        )
        REPORT_PATH.write_text("\n".join(report_lines))
        print("No MXI futures history downloaded.")
        print(f"Saved report to {REPORT_PATH}")
        return 1

    history = pd.concat(history_frames, ignore_index=True, sort=False)
    if "TRADEDATE" in history.columns:
        history["TRADEDATE"] = pd.to_datetime(
            history["TRADEDATE"], errors="coerce"
        ).dt.normalize()

    for col in ["OPEN", "LOW", "HIGH", "CLOSE", "VOLUME", "VALUE", "OPENPOSITION"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    subset = ["SECID", "TRADEDATE"] if "SECID" in history.columns else ["SECID_REQUESTED", "TRADEDATE"]
    history = history.drop_duplicates(subset=subset).sort_values(subset).reset_index(drop=True)

    history.to_parquet(HISTORY_PARQUET_PATH, index=False)
    history.to_csv(HISTORY_CSV_PATH, index=False)

    report_lines.extend(
        [
            f"metadata rows: {len(metadata)}",
            f"history rows: {len(history)}",
            "history columns: " + ", ".join(history.columns.astype(str).tolist()),
            "history SECIDs: "
            + ", ".join(
                sorted(
                    history["SECID"].astype(str).dropna().unique().tolist()
                    if "SECID" in history.columns
                    else history["SECID_REQUESTED"].astype(str).dropna().unique().tolist()
                )
            ),
        ]
    )

    if "TRADEDATE" in history.columns and history["TRADEDATE"].notna().any():
        report_lines.extend(
            [
                f"min TRADEDATE: {history['TRADEDATE'].min().date()}",
                f"max TRADEDATE: {history['TRADEDATE'].max().date()}",
                "rows by year:",
                history["TRADEDATE"].dt.year.value_counts().sort_index().to_string(),
            ]
        )

    report_lines.extend(
        [
            f"failed SECIDs: {len(failed)}",
            *failed,
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines))

    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Saved history to {HISTORY_PARQUET_PATH}")
    print(f"Saved history to {HISTORY_CSV_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print("shape:", history.shape)
    if "TRADEDATE" in history.columns and history["TRADEDATE"].notna().any():
        print("min date:", history["TRADEDATE"].min())
        print("max date:", history["TRADEDATE"].max())
    print("NaN stats:")
    print(history.isna().mean().sort_values(ascending=False).head(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
