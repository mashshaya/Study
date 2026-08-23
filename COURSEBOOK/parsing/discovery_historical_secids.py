#!/usr/bin/env python3
"""Discover historical candidate SECIDs for IMOEX/MIX/MXI options."""

from __future__ import annotations

import time

import pandas as pd

from HSE.COURSEBOOK.parsing.imoex_options_pipeline_common import (
    CANDIDATES_CSV,
    CANDIDATES_PARQUET,
    DISCOVERY_BASE_QUERIES,
    DISCOVERY_REPORT,
    SLEEP_SECONDS,
    add_parsed_contract_fields,
    create_session,
    ensure_output_dir,
    fetch_search_results,
    generate_queries_from_row,
    is_option_like_row,
    save_dataframe,
    setup_logger,
    write_text_report,
)


MAX_EXPANSION_ROWS = 40


def main() -> int:
    ensure_output_dir()
    logger = setup_logger("discovery_historical_secids")
    session = create_session()

    base_frames: list[pd.DataFrame] = []
    base_log: list[tuple[str, int, str | None]] = []

    for query in DISCOVERY_BASE_QUERIES:
        frame, error = fetch_search_results(session, query, logger)
        base_log.append((query, len(frame), error))
        if not frame.empty:
            frame = frame.copy()
            frame["_query"] = query
            frame["_strategy"] = "base_query"
            base_frames.append(frame)
        time.sleep(SLEEP_SECONDS)

    if base_frames:
        base_df = pd.concat(base_frames, ignore_index=True)
        base_df = base_df.drop_duplicates().reset_index(drop=True)
    else:
        base_df = pd.DataFrame()

    filtered_base_df = (
        base_df.loc[base_df.apply(is_option_like_row, axis=1)].copy()
        if not base_df.empty
        else pd.DataFrame()
    )

    expansion_frames: list[pd.DataFrame] = []
    expansion_log: list[tuple[str, str, int, str | None]] = []

    if not filtered_base_df.empty:
        for _, row in filtered_base_df.head(MAX_EXPANSION_ROWS).iterrows():
            for strategy, query in generate_queries_from_row(row):
                frame, error = fetch_search_results(session, query, logger)
                expansion_log.append((strategy, query, len(frame), error))
                if not frame.empty:
                    frame = frame.copy()
                    frame["_query"] = query
                    frame["_strategy"] = strategy
                    expansion_frames.append(frame)
                time.sleep(SLEEP_SECONDS)

    if expansion_frames:
        expansion_df = pd.concat(expansion_frames, ignore_index=True)
        expansion_df = expansion_df.drop_duplicates().reset_index(drop=True)
    else:
        expansion_df = pd.DataFrame()

    all_frames = [frame for frame in [base_df, expansion_df] if not frame.empty]
    if all_frames:
        combined_df = pd.concat(all_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    if not combined_df.empty:
        dedupe_cols = [
            column
            for column in ["secid", "shortname", "name", "type", "group"]
            if column in combined_df.columns
        ]
        if dedupe_cols:
            combined_df = combined_df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
        combined_df = combined_df.loc[combined_df.apply(is_option_like_row, axis=1)].copy()
        combined_df = add_parsed_contract_fields(combined_df)
        if "secid" in combined_df.columns:
            combined_df = combined_df.dropna(subset=["secid"]).reset_index(drop=True)
            combined_df["secid"] = combined_df["secid"].astype(str)
    else:
        combined_df = add_parsed_contract_fields(pd.DataFrame())

    save_dataframe(combined_df, CANDIDATES_PARQUET, CANDIDATES_CSV)

    lines = [
        "Discovery historical SECIDs report",
        "",
        f"base queries tested: {len(base_log)}",
        "base query results:",
    ]
    for query, rows, error in base_log:
        line = f"- {query!r}: rows={rows}"
        if error:
            line += f", error={error}"
        lines.append(line)

    lines.extend(
        [
            "",
            f"base raw rows: {len(base_df)}",
            f"base filtered rows: {len(filtered_base_df)}",
            f"expansion query count: {len(expansion_log)}",
            "expansion query results:",
        ]
    )
    for strategy, query, rows, error in expansion_log:
        line = f"- {strategy}: {query!r} -> rows={rows}"
        if error:
            line += f", error={error}"
        lines.append(line)

    lines.extend(
        [
            "",
            f"expansion raw rows: {len(expansion_df)}",
            f"final candidate rows: {len(combined_df)}",
            f"unique SECIDs: {combined_df['secid'].nunique() if 'secid' in combined_df.columns else 0}",
            f"output parquet: {CANDIDATES_PARQUET}",
        ]
    )

    if not combined_df.empty and "secid" in combined_df.columns:
        lines.extend(["", "first 30 candidate SECIDs:"])
        for secid in combined_df["secid"].head(30).tolist():
            lines.append(f"- {secid}")

    write_text_report(DISCOVERY_REPORT, lines)
    logger.info("Saved candidates to %s", CANDIDATES_PARQUET)
    logger.info("Saved report to %s", DISCOVERY_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
