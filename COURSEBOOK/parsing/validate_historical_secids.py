#!/usr/bin/env python3
"""Validate candidate SECIDs by checking non-empty history on MOEX ISS."""

from __future__ import annotations

import time

import pandas as pd

from HSE.COURSEBOOK.parsing.imoex_options_pipeline_common import (
    CANDIDATES_PARQUET,
    DATE_FROM,
    DATE_TILL,
    SLEEP_SECONDS,
    VALIDATED_CSV,
    VALIDATED_PARQUET,
    VALIDATION_REPORT,
    create_session,
    ensure_output_dir,
    fetch_full_history,
    save_dataframe,
    setup_logger,
    summarize_history_frame,
    write_text_report,
)


def main() -> int:
    ensure_output_dir()
    logger = setup_logger("validate_historical_secids")
    session = create_session()

    candidates_df = pd.read_parquet(CANDIDATES_PARQUET)
    if candidates_df.empty or "secid" not in candidates_df.columns:
        save_dataframe(pd.DataFrame({"secid": []}), VALIDATED_PARQUET, VALIDATED_CSV)
        write_text_report(
            VALIDATION_REPORT,
            [
                "Validation report",
                "",
                f"candidate source: {CANDIDATES_PARQUET}",
                "No candidate SECIDs found.",
            ],
        )
        logger.warning("No candidate SECIDs to validate.")
        return 0

    validated_rows: list[dict[str, object]] = []
    empty_secids: list[str] = []
    failed_secids: list[tuple[str, str]] = []

    for _, row in candidates_df.drop_duplicates(subset=["secid"]).iterrows():
        secid = str(row["secid"])
        history_df, error = fetch_full_history(session, secid, logger)
        if error:
            failed_secids.append((secid, error))
        elif history_df.empty:
            empty_secids.append(secid)
        else:
            summary = summarize_history_frame(history_df)
            history_dates = pd.to_datetime(history_df["TRADEDATE"], errors="coerce")
            years = history_dates.dt.year.dropna().astype(int)
            row_dict = row.to_dict()
            row_dict["history_rows"] = summary["rows"]
            row_dict["first_tradedate"] = summary["first_tradedate"]
            row_dict["last_tradedate"] = summary["last_tradedate"]
            row_dict["rows_2024"] = int((years == 2024).sum())
            row_dict["rows_2025"] = int((years == 2025).sum())
            row_dict["rows_2026"] = int((years == 2026).sum())
            validated_rows.append(row_dict)
        time.sleep(SLEEP_SECONDS)

    validated_df = pd.DataFrame(validated_rows)
    if not validated_df.empty:
        validated_df = validated_df.sort_values(
            ["history_rows", "secid"], ascending=[False, True]
        ).reset_index(drop=True)

    save_dataframe(validated_df, VALIDATED_PARQUET, VALIDATED_CSV)

    lines = [
        "Validation report",
        "",
        f"candidate source: {CANDIDATES_PARQUET}",
        f"target date range: {DATE_FROM} -> {DATE_TILL}",
        f"candidate rows input: {len(candidates_df)}",
        f"unique candidate SECIDs: {candidates_df['secid'].astype(str).nunique()}",
        f"validated SECIDs with non-empty history: {len(validated_df)}",
        f"empty-history SECIDs: {len(empty_secids)}",
        f"failed SECIDs: {len(failed_secids)}",
        "",
        "top validated SECIDs:",
    ]

    if not validated_df.empty:
        preview_cols = [
            col
            for col in [
                "secid",
                "shortname",
                "history_rows",
                "first_tradedate",
                "last_tradedate",
                "rows_2024",
                "rows_2025",
                "rows_2026",
            ]
            if col in validated_df.columns
        ]
        for row in validated_df[preview_cols].head(30).to_dict(orient="records"):
            lines.append("- " + str(row))
    else:
        lines.append("- none")

    lines.extend(["", "empty-history SECIDs:"])
    if empty_secids:
        lines.extend(f"- {secid}" for secid in empty_secids[:100])
    else:
        lines.append("- none")

    lines.extend(["", "failed SECIDs:"])
    if failed_secids:
        lines.extend(f"- {secid}: {error}" for secid, error in failed_secids)
    else:
        lines.append("- none")

    write_text_report(VALIDATION_REPORT, lines)
    logger.info("Saved validated SECIDs to %s", VALIDATED_PARQUET)
    logger.info("Saved report to %s", VALIDATION_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
