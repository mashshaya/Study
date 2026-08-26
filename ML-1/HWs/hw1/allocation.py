from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CAPACITIES = {
    "Количественные финансы": 60,
    "Промышленное программирование на Haskell": 60,
    "Рекомендательные системы": 60,
    "Глубинное обучение в обработке звука": 1000,
}
DEFAULT_CAPACITY = 30
ML2_COURSE = "Машинное обучение 2"


@dataclass(frozen=True)
class AllocationResult:
    first_wave: pd.DataFrame
    first_wave_flags: pd.DataFrame
    final: pd.DataFrame
    final_long: pd.DataFrame


def is_senior_mop_student(df: pd.DataFrame) -> pd.Series:
    """Return True for fourth-year MOP students who choose two elective courses."""
    return df["group_21"].isin([211, 212, 213])


def is_mop_student(df: pd.DataFrame) -> pd.Series:
    """Return True for students known to be from the MOP specialization."""
    return df["group_21"].isin([211, 212, 213]) | df["is_ml_student"].eq(1)


def add_course_counts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["fall_courses"] = 1
    result["spring_courses"] = 1
    result.loc[is_senior_mop_student(result), ["fall_courses", "spring_courses"]] = 2
    return result


def normalize_spring_priorities(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    spring_cols = ["spring_1", "spring_2", "spring_3"]

    for idx in result.index[is_mop_student(result)]:
        priorities = [
            course
            for course in result.loc[idx, spring_cols].tolist()
            if _clean_course(course) and _clean_course(course) != ML2_COURSE
        ]
        priorities = priorities[:3] + [""] * (3 - len(priorities))
        result.loc[idx, spring_cols] = priorities[:3]

    return result


def allocate_courses(
    df: pd.DataFrame,
    prefix: str,
    count_column: str,
    capacities: dict[str, int] | None = None,
    default_capacity: int = DEFAULT_CAPACITY,
) -> AllocationResult:
    capacities = CAPACITIES if capacities is None else capacities
    priority_cols = [f"{prefix}_{i}" for i in range(1, 4)]
    base = df[["id", "percentile", count_column, *priority_cols]].copy()
    base["id"] = base["id"].astype(str)

    assigned = pd.DataFrame(columns=["id", "course", "wave", "percentile"])
    remaining = _initial_remaining(base, priority_cols, capacities, default_capacity)

    first_wave_detail = pd.DataFrame()
    for wave in (1, 2, 3):
        candidates = _wave_candidates(base, assigned, priority_cols, count_column, wave)
        chosen = _select_by_capacity(candidates, remaining)
        if wave == 1:
            first_wave_detail = chosen.copy()

        if not chosen.empty:
            assigned = pd.concat([assigned, chosen], ignore_index=True)
        used = chosen.groupby("course").size()
        remaining = (remaining - used).fillna(remaining).clip(lower=0).astype(int)

    first_wave = _to_wide(first_wave_detail, base, count_column)
    final = _to_wide(assigned, base, count_column)
    flags = _first_wave_flags(first_wave_detail)

    return AllocationResult(
        first_wave=first_wave,
        first_wave_flags=flags,
        final=final,
        final_long=assigned,
    )


def _clean_course(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value in {"", "0", "nan"} else value


def _initial_remaining(
    base: pd.DataFrame,
    priority_cols: list[str],
    capacities: dict[str, int],
    default_capacity: int,
) -> pd.Series:
    courses = (
        base[priority_cols]
        .stack()
        .map(_clean_course)
    )
    courses = sorted(courses[courses.ne("")].unique())
    return pd.Series(
        {course: capacities.get(course, default_capacity) for course in courses},
        dtype=int,
    )


def _wave_candidates(
    base: pd.DataFrame,
    assigned: pd.DataFrame,
    priority_cols: list[str],
    count_column: str,
    wave: int,
) -> pd.DataFrame:
    assigned_count = assigned.groupby("id").size() if len(assigned) else pd.Series(dtype=int)
    need = base["id"].map(assigned_count).fillna(0).astype(int).lt(base[count_column])

    if wave == 1:
        pieces = [
            base.loc[base[count_column].eq(1), ["id", "percentile", priority_cols[0]]]
            .rename(columns={priority_cols[0]: "course"}),
            base.loc[base[count_column].eq(2), ["id", "percentile", priority_cols[0]]]
            .rename(columns={priority_cols[0]: "course"}),
            base.loc[base[count_column].eq(2), ["id", "percentile", priority_cols[1]]]
            .rename(columns={priority_cols[1]: "course"}),
        ]
        candidates = pd.concat(pieces, ignore_index=True)
    elif wave == 2:
        candidates = pd.concat(
            [
                base.loc[base[count_column].eq(1) & need, ["id", "percentile", priority_cols[1]]]
                .rename(columns={priority_cols[1]: "course"}),
                base.loc[base[count_column].eq(2) & need, ["id", "percentile", priority_cols[2]]]
                .rename(columns={priority_cols[2]: "course"}),
            ],
            ignore_index=True,
        )
    else:
        candidates = base.loc[need, ["id", "percentile", priority_cols[2]]].rename(
            columns={priority_cols[2]: "course"}
        )

    candidates["course"] = candidates["course"].map(_clean_course)
    candidates = candidates[candidates["course"].ne("")]

    if len(assigned):
        taken = assigned[["id", "course"]].drop_duplicates()
        candidates = candidates.merge(taken, on=["id", "course"], how="left", indicator=True)
        candidates = candidates[candidates["_merge"].eq("left_only")].drop(columns="_merge")

    candidates["wave"] = wave
    return candidates


def _select_by_capacity(candidates: pd.DataFrame, remaining: pd.Series) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    ranked = candidates.sort_values(["course", "percentile"], ascending=[True, True]).copy()
    ranked["remaining"] = ranked["course"].map(remaining).fillna(0).astype(int)
    ranked["rank"] = ranked.groupby("course").cumcount()
    return ranked.loc[ranked["rank"].lt(ranked["remaining"]), ["id", "course", "wave", "percentile"]]


def _first_wave_flags(first_wave_detail: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(columns=["id", "is_first_place", "is_last_place"])
    if first_wave_detail.empty:
        return flags

    detail = first_wave_detail.copy()
    detail["is_first_place"] = detail["percentile"].eq(detail.groupby("course")["percentile"].transform("min"))
    detail["is_last_place"] = detail["percentile"].eq(detail.groupby("course")["percentile"].transform("max"))
    flags = detail.groupby("id")[["is_first_place", "is_last_place"]].max().reset_index()
    flags["is_first_place"] = flags["is_first_place"].where(flags["is_first_place"])
    flags["is_last_place"] = flags["is_last_place"].where(flags["is_last_place"])
    return flags


def _to_wide(assigned: pd.DataFrame, base: pd.DataFrame, count_column: str) -> pd.DataFrame:
    result = base[["id", count_column]].copy()
    courses = assigned.sort_values(["id", "wave", "course"]) if len(assigned) else assigned

    if len(courses):
        numbered = courses[["id", "course"]].copy()
        numbered["course_num"] = numbered.groupby("id").cumcount() + 1
        wide = numbered.pivot(index="id", columns="course_num", values="course").reset_index()
        wide = wide.rename(columns={1: "course_1", 2: "course_2"})
        result = result.merge(wide, on="id", how="left")
    else:
        result["course_1"] = pd.NA
        result["course_2"] = pd.NA

    if "course_1" not in result:
        result["course_1"] = pd.NA
    if "course_2" not in result:
        result["course_2"] = pd.NA

    result.loc[result[count_column].eq(1), "course_1"] = result.loc[
        result[count_column].eq(1), "course_1"
    ].fillna("???")
    result.loc[result[count_column].eq(1), "course_2"] = "-"
    result.loc[result[count_column].eq(2), "course_1"] = result.loc[
        result[count_column].eq(2), "course_1"
    ].fillna("???")
    result.loc[result[count_column].eq(2), "course_2"] = result.loc[
        result[count_column].eq(2), "course_2"
    ].fillna("???")

    return result[["id", "course_1", "course_2"]]
