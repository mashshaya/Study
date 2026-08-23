from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

MONTHS = {
    "январь": 1,
    "января": 1,
    "февраль": 2,
    "февраля": 2,
    "март": 3,
    "марта": 3,
    "апрель": 4,
    "апреля": 4,
    "май": 5,
    "мая": 5,
    "июнь": 6,
    "июня": 6,
    "июль": 7,
    "июля": 7,
    "август": 8,
    "августа": 8,
    "сентябрь": 9,
    "сентября": 9,
    "октябрь": 10,
    "октября": 10,
    "ноябрь": 11,
    "ноября": 11,
    "декабрь": 12,
    "декабря": 12,
}


def _prepare_xlrd() -> None:
    vendor_dir = Path("/private/tmp/codex_xlrd")
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))


_prepare_xlrd()
import xlrd  # type: ignore  # noqa: E402


@dataclass
class SeriesSpec:
    name: str
    series: pd.Series
    source_file: str
    frequency: str
    transform_note: str


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def _open_sheet(path: Path):
    workbook = xlrd.open_workbook(str(path))
    return workbook.sheet_by_name("Данные")


def _month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1)


def load_m2_series(path: Path) -> SeriesSpec:
    sheet = _open_sheet(path)
    current_year: int | None = None
    rows = []
    for col in range(2, sheet.ncols):
        year_value = sheet.cell_value(2, col)
        if year_value != "":
            current_year = int(year_value)
        month_name = str(sheet.cell_value(3, col)).strip().lower()
        value = sheet.cell_value(4, col)
        if current_year and month_name in MONTHS and _is_number(value):
            rows.append((_month_start(current_year, MONTHS[month_name]), float(value)))
    series = pd.Series(
        data=[value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows], name="date"),
        name="m2_bln_rub",
    ).sort_index()
    return SeriesSpec(
        name="m2_bln_rub",
        series=series,
        source_file=path.name,
        frequency="monthly",
        transform_note="Raw monthly M2 series.",
    )


def load_cpi_series(path: Path) -> SeriesSpec:
    sheet = _open_sheet(path)
    current_year: int | None = None
    rows = []
    for col in range(1, sheet.ncols):
        year_value = sheet.cell_value(2, col)
        if year_value != "":
            current_year = int(year_value)
        month_name = str(sheet.cell_value(3, col)).strip().lower()
        value = sheet.cell_value(4, col)
        if current_year and month_name in MONTHS and _is_number(value):
            rows.append((_month_start(current_year, MONTHS[month_name]), float(value)))
    series = pd.Series(
        data=[value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows], name="date"),
        name="cpi_all_items_index_2010_100",
    ).sort_index()
    return SeriesSpec(
        name="cpi_all_items_index_2010_100",
        series=series,
        source_file=path.name,
        frequency="monthly",
        transform_note="Raw monthly CPI index level for all goods and services.",
    )


def load_usd_series(path: Path) -> SeriesSpec:
    sheet = _open_sheet(path)
    rows = []
    for row in range(3, sheet.nrows):
        year_value = sheet.cell_value(row, 0)
        month_name = str(sheet.cell_value(row, 1)).strip().lower()
        value = sheet.cell_value(row, 2)
        if isinstance(year_value, str) and year_value.strip().isdigit():
            year_value = int(year_value.strip())
        if _is_number(year_value) and month_name in MONTHS and _is_number(value):
            rows.append((_month_start(int(year_value), MONTHS[month_name]), float(value)))
    series = pd.Series(
        data=[value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows], name="date"),
        name="usd_rub_avg",
    ).sort_index()
    return SeriesSpec(
        name="usd_rub_avg",
        series=series,
        source_file=path.name,
        frequency="monthly",
        transform_note="Monthly average USD/RUB exchange rate.",
    )


def _parse_russian_date(year: int, text: str) -> pd.Timestamp:
    day_text, month_text = text.strip().lower().split()
    return pd.Timestamp(year=year, month=MONTHS[month_text], day=int(day_text))


def load_key_rate_series(path: Path) -> SeriesSpec:
    sheet = _open_sheet(path)
    events = []
    current_year: int | None = None
    for col in range(2, sheet.ncols):
        year_value = sheet.cell_value(2, col)
        if year_value != "":
            current_year = int(year_value)
        date_text = str(sheet.cell_value(3, col)).strip()
        value = sheet.cell_value(4, col)
        if current_year and date_text and _is_number(value):
            events.append((_parse_russian_date(current_year, date_text), float(value)))

    event_frame = pd.DataFrame(events, columns=["effective_date", "key_rate_pct"]).sort_values("effective_date")
    month_index = pd.date_range(
        event_frame["effective_date"].min().to_period("M").to_timestamp(),
        event_frame["effective_date"].max().to_period("M").to_timestamp(),
        freq="MS",
    )
    monthly_frame = pd.DataFrame({"date": month_index})
    monthly_frame["month_end"] = monthly_frame["date"] + pd.offsets.MonthEnd(0)
    aligned = pd.merge_asof(
        monthly_frame.sort_values("month_end"),
        event_frame,
        left_on="month_end",
        right_on="effective_date",
        direction="backward",
    )
    series = pd.Series(
        data=aligned["key_rate_pct"].to_list(),
        index=pd.DatetimeIndex(aligned["date"], name="date"),
        name="key_rate_pct",
    )
    return SeriesSpec(
        name="key_rate_pct",
        series=series,
        source_file=path.name,
        frequency="monthly_from_irregular",
        transform_note="Converted from decision dates to monthly series using last observed rate within each month (forward fill between CB meetings).",
    )


def build_all_series() -> dict[str, SeriesSpec]:
    return {
        "m2_bln_rub": load_m2_series(DATA_DIR / "data (21).xls"),
        "cpi_all_items_index_2010_100": load_cpi_series(DATA_DIR / "data (22).xls"),
        "usd_rub_avg": load_usd_series(DATA_DIR / "data (23).xls"),
        "key_rate_pct": load_key_rate_series(DATA_DIR / "data (24).xls"),
    }


def merge_selected_series(specs: dict[str, SeriesSpec], how: str = "outer") -> pd.DataFrame:
    frame = pd.concat([spec.series for spec in specs.values()], axis=1, join=how).sort_index()
    frame.index.name = "date"
    return frame.reset_index()


def make_overlap_report(specs: dict[str, SeriesSpec], min_obs: int = 26) -> dict[str, object]:
    overlap_frame = merge_selected_series(specs, how="inner")
    pairwise = []
    keys = list(specs.keys())
    for i, left_key in enumerate(keys):
        for right_key in keys[i + 1 :]:
            pair = pd.concat([specs[left_key].series, specs[right_key].series], axis=1, join="inner").dropna()
            pairwise.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "shared_observations": int(len(pair)),
                    "start": pair.index.min().strftime("%Y-%m-%d") if not pair.empty else None,
                    "end": pair.index.max().strftime("%Y-%m-%d") if not pair.empty else None,
                }
            )

    metadata = []
    for spec in specs.values():
        metadata.append(
            {
                "series": spec.name,
                "source_file": spec.source_file,
                "frequency": spec.frequency,
                "start": spec.series.index.min().strftime("%Y-%m-%d"),
                "end": spec.series.index.max().strftime("%Y-%m-%d"),
                "observations": int(spec.series.notna().sum()),
                "transform_note": spec.transform_note,
            }
        )

    return {
        "series_metadata": metadata,
        "pairwise_overlap": pairwise,
        "strict_intersection_observations": int(len(overlap_frame)),
        "strict_intersection_start": overlap_frame["date"].min().strftime("%Y-%m-%d") if not overlap_frame.empty else None,
        "strict_intersection_end": overlap_frame["date"].max().strftime("%Y-%m-%d") if not overlap_frame.empty else None,
        "passes_assignment_threshold": bool(len(overlap_frame) >= min_obs),
        "threshold_observations_required": min_obs,
    }


def save_outputs() -> dict[str, object]:
    specs = build_all_series()
    OUTPUT_DIR.mkdir(exist_ok=True)

    outer_frame = merge_selected_series(specs, how="outer")
    inner_frame = merge_selected_series(specs, how="inner")
    report = make_overlap_report(specs)

    outer_frame.to_csv(OUTPUT_DIR / "selected_series_outer_monthly.csv", index=False)
    inner_frame.to_csv(OUTPUT_DIR / "selected_series_strict_intersection.csv", index=False)
    with open(OUTPUT_DIR / "selected_series_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    report = save_outputs()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passes_assignment_threshold"]:
        print(
            "\nResult: the selected four series do not produce a final common monthly dataset with more than 25 observations."
        )


if __name__ == "__main__":
    main()
