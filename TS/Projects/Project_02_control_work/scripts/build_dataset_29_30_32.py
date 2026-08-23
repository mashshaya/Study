from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TARGET_INDEX = pd.date_range("2013-01-01", "2015-06-01", freq="MS")

MONTHS = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}


def _prepare_xlrd() -> None:
    vendor_dir = Path("/private/tmp/codex_xlrd")
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))


_prepare_xlrd()
import xlrd  # type: ignore  # noqa: E402


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def _open_sheet(path: Path):
    workbook = xlrd.open_workbook(str(path))
    return workbook.sheet_by_name("Данные")


def _month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1)


def load_cpi_all_items() -> pd.Series:
    sheet = _open_sheet(DATA_DIR / "data (29).xls")
    current_year: int | None = None
    rows: list[tuple[pd.Timestamp, float]] = []
    for col in range(1, sheet.ncols):
        year_value = sheet.cell_value(2, col)
        if year_value != "":
            current_year = int(year_value)
        month_name = str(sheet.cell_value(3, col)).strip().lower()
        value = sheet.cell_value(4, col)
        if current_year and month_name in MONTHS and _is_number(value):
            rows.append((_month_start(current_year, MONTHS[month_name]), float(value)))
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows], name="date"),
        name="cpi_all_items_index_2010_100",
    ).sort_index()


def load_net_foreign_assets() -> pd.Series:
    sheet = _open_sheet(DATA_DIR / "data (30).xls")
    current_year: int | None = None
    rows: list[tuple[pd.Timestamp, float]] = []
    for col in range(3, sheet.ncols):
        year_value = sheet.cell_value(2, col)
        if year_value != "":
            current_year = int(year_value)
        month_name = str(sheet.cell_value(3, col)).strip().lower()
        value = sheet.cell_value(4, col)
        if current_year and month_name in MONTHS and _is_number(value):
            rows.append((_month_start(current_year, MONTHS[month_name]), float(value) / 1000.0))
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows], name="date"),
        name="net_foreign_assets_bln_rub",
    ).sort_index()


def load_m2() -> pd.Series:
    sheet = _open_sheet(DATA_DIR / "data (32).xls")
    current_year: int | None = None
    rows: list[tuple[pd.Timestamp, float]] = []
    for col in range(2, sheet.ncols):
        year_value = sheet.cell_value(2, col)
        if year_value != "":
            current_year = int(year_value)
        month_name = str(sheet.cell_value(3, col)).strip().lower()
        value = sheet.cell_value(4, col)
        if current_year and month_name in MONTHS and _is_number(value):
            rows.append((_month_start(current_year, MONTHS[month_name]), float(value)))
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows], name="date"),
        name="m2_bln_rub",
    ).sort_index()


def build_dataset() -> pd.DataFrame:
    frame = pd.concat(
        [
            load_cpi_all_items(),
            load_net_foreign_assets(),
            load_m2(),
        ],
        axis=1,
        join="outer",
    ).sort_index()
    frame = frame.reindex(TARGET_INDEX)
    frame["net_foreign_assets_bln_rub"] = frame["net_foreign_assets_bln_rub"].ffill()
    frame = frame.loc[:, ["cpi_all_items_index_2010_100", "net_foreign_assets_bln_rub", "m2_bln_rub"]]
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def build_report(dataset: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset_name": "final_dataset_29_30_32",
        "series": [
            {
                "column": "cpi_all_items_index_2010_100",
                "source_file": "data (29).xls",
                "description": "Consumer price index for all goods and services.",
                "unit": "index, 2010 average = 100",
            },
            {
                "column": "net_foreign_assets_bln_rub",
                "source_file": "data (30).xls",
                "description": "Net foreign assets of credit organizations.",
                "unit": "billion rubles",
            },
            {
                "column": "m2_bln_rub",
                "source_file": "data (32).xls",
                "description": "Money aggregate M2.",
                "unit": "billion rubles",
            },
        ],
        "start_date": dataset["date"].min(),
        "end_date": dataset["date"].max(),
        "observations": int(len(dataset)),
        "columns": dataset.columns.tolist(),
        "missing_values": {column: int(dataset[column].isna().sum()) for column in dataset.columns if column != "date"},
        "assumption": "Net foreign assets are forward-filled from 2014-12 through 2015-06 to extend the common sample.",
    }


def save_outputs() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = build_dataset()
    report = build_report(dataset)

    csv_path = OUTPUT_DIR / "final_dataset_29_30_32.csv"
    report_path = OUTPUT_DIR / "final_dataset_29_30_32_report.json"

    dataset.to_csv(csv_path, index=False)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return csv_path, report_path


def main() -> None:
    csv_path, report_path = save_outputs()
    dataset = pd.read_csv(csv_path)
    print(f"Saved dataset: {csv_path}")
    print(f"Saved report: {report_path}")
    print(dataset.head().to_string(index=False))
    print()
    print(dataset.tail().to_string(index=False))


if __name__ == "__main__":
    main()
