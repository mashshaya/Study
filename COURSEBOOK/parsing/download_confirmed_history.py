#!/usr/bin/env python3
"""Download full 2024-2026 history for validated IMOEX option SECIDs."""

from __future__ import annotations

from pathlib import Path
import json
import time

import pandas as pd

from HSE.COURSEBOOK.parsing.imoex_options_pipeline_common import (
    COVERAGE_BY_YEAR_CSV,
    DATE_FROM,
    DATE_TILL,
    DOWNLOAD_REPORT,
    FINAL_HISTORY_CSV,
    FINAL_HISTORY_PARQUET,
    FINAL_RAW_COLUMNS,
    MISSING_SHARE_CSV,
    QUALITY_SUMMARY_JSON,
    ROWS_BY_SECID_CSV,
    SLEEP_SECONDS,
    VALIDATED_PARQUET,
    create_session,
    ensure_output_dir,
    fetch_full_history,
    save_dataframe,
    setup_logger,
    summarize_history_frame,
    write_text_report,
)


def build_empty_final_frame() -> pd.DataFrame:
    columns = FINAL_RAW_COLUMNS + ["option_type", "strike", "expiry_date"]
    return pd.DataFrame(columns=columns)


def main() -> int:
    ensure_output_dir()
    logger = setup_logger("download_confirmed_history")
    session = create_session()

    validated_df = pd.read_parquet(VALIDATED_PARQUET)
    if validated_df.empty or "secid" not in validated_df.columns:
        empty_df = build_empty_final_frame()
        save_dataframe(empty_df, FINAL_HISTORY_PARQUET, FINAL_HISTORY_CSV)
        save_dataframe(pd.DataFrame({"year": [], "rows": [], "unique_secids": []}), COVERAGE_BY_YEAR_CSV)
        save_dataframe(pd.DataFrame({"SECID": [], "rows": [], "first_tradedate": [], "last_tradedate": []}), ROWS_BY_SECID_CSV)
        save_dataframe(pd.DataFrame({"column": [], "missing_share": []}), MISSING_SHARE_CSV)
        Path(QUALITY_SUMMARY_JSON).write_text(
            json.dumps({"total_rows": 0, "unique_secids": 0}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_text_report(
            DOWNLOAD_REPORT,
            [
                "Download report",
                "",
                f"validated source: {VALIDATED_PARQUET}",
                "No validated SECIDs found.",
            ],
        )
        logger.warning("No validated SECIDs to download.")
        return 0

    history_frames: list[pd.DataFrame] = []
    empty_secids: list[str] = []
    failed_secids: list[tuple[str, str]] = []

    metadata_cols = [
        col for col in ["secid", "option_type", "strike", "expiry_date"] if col in validated_df.columns
    ]
    metadata_df = validated_df[metadata_cols].drop_duplicates(subset=["secid"]).copy()

    for secid in metadata_df["secid"].astype(str).tolist():
        history_df, error = fetch_full_history(session, secid, logger)
        if error:
            failed_secids.append((secid, error))
        elif history_df.empty:
            empty_secids.append(secid)
        else:
            history_frames.append(history_df)
        time.sleep(SLEEP_SECONDS)

    if history_frames:
        combined_history = pd.concat(history_frames, ignore_index=True)
        combined_history = combined_history.drop_duplicates(
            subset=["SECID", "TRADEDATE"]
        ).reset_index(drop=True)
        combined_history["TRADEDATE"] = pd.to_datetime(
            combined_history["TRADEDATE"], errors="coerce"
        )
        combined_history = combined_history.sort_values(
            ["SECID", "TRADEDATE"], kind="stable"
        ).reset_index(drop=True)
    else:
        combined_history = pd.DataFrame(columns=FINAL_RAW_COLUMNS)

    if not combined_history.empty:
        final_df = combined_history.reindex(columns=FINAL_RAW_COLUMNS).copy()
        final_df = final_df.merge(
            metadata_df.rename(columns={"secid": "SECID"}),
            on="SECID",
            how="left",
        )
    else:
        final_df = build_empty_final_frame()

    save_dataframe(final_df, FINAL_HISTORY_PARQUET, FINAL_HISTORY_CSV)

    if not final_df.empty:
        trade_year = pd.to_datetime(final_df["TRADEDATE"], errors="coerce").dt.year
        coverage_by_year = (
            pd.DataFrame({"year": trade_year, "SECID": final_df["SECID"]})
            .dropna(subset=["year"])
            .assign(year=lambda df: df["year"].astype(int))
            .groupby("year", as_index=False)
            .agg(rows=("SECID", "size"), unique_secids=("SECID", "nunique"))
            .sort_values("year")
            .reset_index(drop=True)
        )
        rows_by_secid = (
            final_df.groupby("SECID", as_index=False)
            .agg(
                rows=("TRADEDATE", "size"),
                first_tradedate=("TRADEDATE", "min"),
                last_tradedate=("TRADEDATE", "max"),
            )
            .sort_values(["rows", "SECID"], ascending=[False, True])
            .reset_index(drop=True)
        )
        missing_share = (
            final_df.isna().mean().rename("missing_share").reset_index().rename(columns={"index": "column"})
        )
        date_range = summarize_history_frame(final_df[["SECID", "TRADEDATE"]])
        quality_summary = {
            "validated_secids_input": int(validated_df["secid"].astype(str).nunique()),
            "downloaded_secids_output": int(final_df["SECID"].astype(str).nunique()),
            "total_rows": int(len(final_df)),
            "date_from_requested": DATE_FROM,
            "date_till_requested": DATE_TILL,
            "first_tradedate": date_range["first_tradedate"],
            "last_tradedate": date_range["last_tradedate"],
            "failed_secids": len(failed_secids),
            "empty_secids": len(empty_secids),
        }
    else:
        coverage_by_year = pd.DataFrame({"year": [], "rows": [], "unique_secids": []})
        rows_by_secid = pd.DataFrame(
            {"SECID": [], "rows": [], "first_tradedate": [], "last_tradedate": []}
        )
        missing_share = pd.DataFrame({"column": [], "missing_share": []})
        quality_summary = {
            "validated_secids_input": int(validated_df["secid"].astype(str).nunique()),
            "downloaded_secids_output": 0,
            "total_rows": 0,
            "date_from_requested": DATE_FROM,
            "date_till_requested": DATE_TILL,
            "first_tradedate": None,
            "last_tradedate": None,
            "failed_secids": len(failed_secids),
            "empty_secids": len(empty_secids),
        }

    coverage_by_year.to_csv(COVERAGE_BY_YEAR_CSV, index=False)
    rows_by_secid.to_csv(ROWS_BY_SECID_CSV, index=False)
    missing_share.to_csv(MISSING_SHARE_CSV, index=False)
    Path(QUALITY_SUMMARY_JSON).write_text(
        json.dumps(quality_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Download report",
        "",
        f"validated source: {VALIDATED_PARQUET}",
        f"requested range: {DATE_FROM} -> {DATE_TILL}",
        f"validated SECIDs input: {validated_df['secid'].astype(str).nunique()}",
        f"downloaded SECIDs output: {final_df['SECID'].astype(str).nunique() if not final_df.empty else 0}",
        f"total rows: {len(final_df)}",
        f"failed SECIDs: {len(failed_secids)}",
        f"empty SECIDs: {len(empty_secids)}",
        f"output parquet: {FINAL_HISTORY_PARQUET}",
        "",
        "coverage by year:",
    ]

    if not coverage_by_year.empty:
        for row in coverage_by_year.to_dict(orient="records"):
            lines.append("- " + str(row))
    else:
        lines.append("- none")

    lines.extend(["", "top rows by SECID:"])
    if not rows_by_secid.empty:
        for row in rows_by_secid.head(30).to_dict(orient="records"):
            lines.append("- " + str(row))
    else:
        lines.append("- none")

    lines.extend(["", "missing share by column:"])
    if not missing_share.empty:
        for row in missing_share.to_dict(orient="records"):
            lines.append("- " + str(row))
    else:
        lines.append("- none")

    lines.extend(["", "failed SECIDs:"])
    if failed_secids:
        for secid, error in failed_secids:
            lines.append(f"- {secid}: {error}")
    else:
        lines.append("- none")

    lines.extend(["", "empty SECIDs:"])
    if empty_secids:
        for secid in empty_secids:
            lines.append(f"- {secid}")
    else:
        lines.append("- none")

    write_text_report(DOWNLOAD_REPORT, lines)
    logger.info("Saved final history to %s", FINAL_HISTORY_PARQUET)
    logger.info("Saved quality artifacts to %s", FINAL_HISTORY_PARQUET.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
