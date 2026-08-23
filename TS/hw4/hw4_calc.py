from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


@dataclass(frozen=True)
class ModelResult:
    name: str
    model: sm.regression.linear_model.RegressionResultsWrapper
    y: pd.Series
    y_hat: pd.Series
    resid: pd.Series


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).reset_index(drop=True)

    # t = 1..T (временной индекс)
    out["t"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _fit_ols(name: str, y: pd.Series, X: pd.DataFrame) -> ModelResult:
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, Xc).fit()
    y_hat = pd.Series(res.fittedvalues, index=y.index, name="y_hat")
    resid = pd.Series(res.resid, index=y.index, name="resid")
    return ModelResult(name=name, model=res, y=y, y_hat=y_hat, resid=resid)


def _metrics(m: ModelResult) -> dict[str, Any]:
    y = m.y.astype(float)
    y_hat = m.y_hat.astype(float)
    resid = (y - y_hat).astype(float)

    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))

    return {
        "model": m.name,
        "n": int(m.model.nobs),
        "k": int(m.model.df_model) + 1,  # incl. intercept
        "R2": float(m.model.rsquared),
        "Adj_R2": float(m.model.rsquared_adj),
        "AIC": float(m.model.aic),
        "BIC": float(m.model.bic),
        "RMSE": rmse,
        "MAE": mae,
    }


def _residual_diagnostics(m: ModelResult, lb_lags: int = 12) -> dict[str, Any]:
    resid = m.resid.astype(float)
    X = m.model.model.exog

    # Durbin–Watson (≈2 хорошо; <2 положит. автокорр.)
    dw = float(sm.stats.stattools.durbin_watson(resid))

    # Ljung–Box: автокорреляция остатков (обычно 12 лагов для месячных данных)
    lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_p = float(lb["lb_pvalue"].iloc[0])

    # Breusch–Pagan: гетероскедастичность
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)

    # Jarque–Bera: нормальность остатков
    jb_stat, jb_p, _, _ = jarque_bera(resid)

    return {
        "model": m.name,
        "DW": dw,
        f"LB({lb_lags})_stat": lb_stat,
        f"LB({lb_lags})_p": lb_p,
        "BP_stat": float(bp_stat),
        "BP_p": float(bp_p),
        "JB_stat": float(jb_stat),
        "JB_p": float(jb_p),
    }


def fit_growth_curves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ModelResult], pd.DataFrame]:
    d = _prepare_df(df)

    # y_t
    y = d["value"]

    # Model 1: linear y_t = b0 + b1 t + e
    m1 = _fit_ols("Linear: y ~ t", y=y, X=d[["t"]])

    # Model 2: quadratic y_t = b0 + b1 t + b2 t^2 + e
    d["t2"] = d["t"] ** 2
    m2 = _fit_ols("Quadratic: y ~ t + t^2", y=y, X=d[["t", "t2"]])

    # Model 3: log-linear ln(y_t) = b0 + b1 t + e  (только если y>0)
    # Для ИПЦ (%) y всегда > 0, но на всякий случай фильтруем.
    d_pos = d[d["value"] > 0].copy()
    y_log = np.log(d_pos["value"])
    m3 = _fit_ols("Log-linear: ln(y) ~ t", y=y_log, X=d_pos[["t"]])

    models: dict[str, ModelResult] = {m.name: m for m in (m1, m2, m3)}

    # Таблица сравнения моделей
    compare = pd.DataFrame([_metrics(m) for m in models.values()]).sort_values(
        by=["AIC", "BIC", "RMSE"], ascending=[True, True, True]
    )

    # Диагностика остатков (для m3 остатки в лог-скейле; это ок, просто так и интерпретируй)
    diagnostics = pd.DataFrame([_residual_diagnostics(m, lb_lags=12) for m in models.values()])

    # Данные для графиков (факт vs прогноз) для m1/m2 на исходном y
    fitted_df = d[["date", "value"]].copy()
    fitted_df["y_hat_linear"] = m1.y_hat.values
    fitted_df["y_hat_quadratic"] = m2.y_hat.values

    # Для log-linear: прогноз на уровне y = exp(ŷ_log)
    fitted_df = fitted_df.merge(d_pos[["date"]], on="date", how="left", indicator=False)
    yhat_log = pd.Series(m3.y_hat, index=d_pos.index)
    pred_level = pd.Series(np.exp(yhat_log), index=d_pos.index)
    tmp = pd.DataFrame({"date": d_pos["date"].values, "y_hat_loglinear_level": pred_level.values})
    fitted_df = fitted_df.merge(tmp, on="date", how="left")

    return compare.reset_index(drop=True), diagnostics.reset_index(drop=True), models, fitted_df

# Optional: печать summary для каждой модели (в ноутбуке)
def print_summaries(models: dict[str, ModelResult]) -> None:
    for name, m in models.items():
        print("\n" + "=" * 80)
        print(name)
        print(m.model.summary())


# Optional: графики (сохраняются в png, чтобы вставить в pdf)
def save_step1_plots(
    fitted: pd.DataFrame,
    models: dict[str, ModelResult],
    out_dir: str = "plots_step1",
) -> None:
    import os

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: исходный ряд
    plt.figure()
    plt.plot(fitted["date"], fitted["value"])
    plt.title("ИПЦ (к предыдущему месяцу), исходный ряд")
    plt.ylabel("%")
    plt.xlabel("date")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/series_raw.png", dpi=200)
    plt.close()

    # Plot 2: факт vs прогнозы (linear/quadratic/log-linear level)
    plt.figure()
    plt.plot(fitted["date"], fitted["value"], label="actual")
    plt.plot(fitted["date"], fitted["y_hat_linear"], label="linear fit")
    plt.plot(fitted["date"], fitted["y_hat_quadratic"], label="quadratic fit")
    plt.plot(fitted["date"], fitted["y_hat_loglinear_level"], label="log-linear fit (level)")
    plt.title("Факт vs прогноз (кривые роста)")
    plt.ylabel("%")
    plt.xlabel("date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/actual_vs_fits.png", dpi=200)
    plt.close()

    # Plot 3: остатки по моделям (m3 в лог-скейле)
    for name, m in models.items():
        # Для log-linear даты берём из fitted через merge по индексу не всегда удобно,
        # поэтому используем fitted для первых двух, а для log-linear построим по d_pos индексу.
        plt.figure()
        if name.startswith("Log-linear"):
            # В log-linear y и resid относятся к d_pos (где y>0)
            # Здесь берём даты из fitted, где есть y_hat_loglinear_level
            df_plot = fitted.dropna(subset=["y_hat_loglinear_level"]).copy()
            # Остатки в log-шкале берём напрямую из m.resid и выравниваем по длине
            resid = pd.Series(m.resid.values, index=df_plot.index)
            plt.plot(df_plot["date"], resid)
            plt.title(f"Остатки: {name} (лог-шкала)")
        else:
            resid = m.resid
            plt.plot(fitted["date"], resid)
            plt.title(f"Остатки: {name}")
        plt.axhline(0.0, linewidth=1)
        plt.ylabel("residual")
        plt.xlabel("date")
        plt.tight_layout()
        safe = (
            name.replace(":", "")
            .replace("~", "")
            .replace(" ", "_")
            .replace("^", "2")
            .replace("(", "")
            .replace(")", "")
        )
        plt.savefig(f"{out_dir}/residuals_{safe}.png", dpi=200)
        plt.close()

TrendSpec = Literal["linear", "quadratic"]

@dataclass(frozen=True)
class ChowResult:
    break_date: pd.Timestamp
    break_index: int
    n_total: int
    n_left: int
    n_right: int
    k_params: int
    f_stat: float
    p_value: float
    rss_pooled: float
    rss_left: float
    rss_right: float


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values("date").reset_index(drop=True)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["date", "value"]).reset_index(drop=True)
    d["t"] = np.arange(1, len(d) + 1, dtype=int)
    d["t2"] = d["t"] ** 2
    return d


def _design(d: pd.DataFrame, spec: TrendSpec) -> pd.DataFrame:
    if spec == "linear":
        X = d[["t"]]
    elif spec == "quadratic":
        X = d[["t", "t2"]]
    else:
        raise ValueError(f"Unknown spec: {spec}")
    return sm.add_constant(X, has_constant="add")


def _fit_rss(y: pd.Series, X: pd.DataFrame) -> float:
    res = sm.OLS(y, X).fit()
    resid = res.resid.astype(float)
    return float(np.sum(resid**2))


def chow_test(
    df: pd.DataFrame,
    break_date: str | pd.Timestamp,
    *,
    spec: TrendSpec = "linear",
    min_segment: int = 24,
) -> ChowResult:
    """
    Chow test at a single break_date.
    Model: y_t = f(t) + e_t, where f(t) is linear/quadratic trend.

    min_segment: minimum observations in each subsample (e.g., 24 months).
    """
    d = _prepare(df)
    bd = pd.Timestamp(break_date)

    # index of last obs in left segment: date <= bd
    left_mask = d["date"] <= bd
    n_left = int(left_mask.sum())
    n_total = int(len(d))
    n_right = n_total - n_left

    if n_left < min_segment or n_right < min_segment:
        raise ValueError(
            f"Break {bd.date()} invalid: left={n_left}, right={n_right}, "
            f"min_segment={min_segment}"
        )

    y = d["value"]
    X = _design(d, spec)
    rss_pooled = _fit_rss(y, X)

    d_left = d.iloc[:n_left].copy()
    d_right = d.iloc[n_left:].copy()

    rss_left = _fit_rss(d_left["value"], _design(d_left, spec))
    rss_right = _fit_rss(d_right["value"], _design(d_right, spec))

    # number of parameters (incl intercept)
    k = int(X.shape[1])

    # Chow F-stat
    num = (rss_pooled - (rss_left + rss_right)) / k
    den = (rss_left + rss_right) / (n_total - 2 * k)
    f_stat = float(num / den)

    # p-value: F(k, n_total - 2k)
    p_value = float(1.0 - stats.f.cdf(f_stat, dfn=k, dfd=(n_total - 2 * k)))

    return ChowResult(
        break_date=bd,
        break_index=n_left,  # split point in 0..n
        n_total=n_total,
        n_left=n_left,
        n_right=n_right,
        k_params=k,
        f_stat=f_stat,
        p_value=p_value,
        rss_pooled=rss_pooled,
        rss_left=rss_left,
        rss_right=rss_right,
    )


def chow_test_many(
    df: pd.DataFrame,
    break_dates: list[str | pd.Timestamp],
    *,
    spec: TrendSpec = "linear",
    min_segment: int = 24,
) -> pd.DataFrame:
    """
    Run Chow test for multiple candidate break dates.
    Returns a tidy table suitable for report.
    """
    rows: list[dict[str, object]] = []
    for bd in break_dates:
        try:
            r = chow_test(df, bd, spec=spec, min_segment=min_segment)
            rows.append(
                {
                    "break_date": r.break_date.date().isoformat(),
                    "spec": spec,
                    "n_left": r.n_left,
                    "n_right": r.n_right,
                    "k": r.k_params,
                    "F": r.f_stat,
                    "p_value": r.p_value,
                    "decision_5%": "reject H0 (break)" if r.p_value < 0.05 else "fail to reject",
                }
            )
        except ValueError as e:
            rows.append(
                {
                    "break_date": pd.Timestamp(bd).date().isoformat(),
                    "spec": spec,
                    "n_left": None,
                    "n_right": None,
                    "k": None,
                    "F": None,
                    "p_value": None,
                    "decision_5%": f"skipped: {e}",
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["p_value", "F"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return out


def default_candidate_breaks() -> list[pd.Timestamp]:
    """
    Reasonable macro break candidates for РФ (monthly):
    2008-09 (global crisis escalation), 2014-12 (FX shock/sanctions),
    2020-04 (COVID shock), 2022-03 (sanctions/inflation shock).
    """
    return [
        pd.Timestamp("2008-09-01"),
        pd.Timestamp("2014-12-01"),
        pd.Timestamp("2020-04-01"),
        pd.Timestamp("2022-03-01"),
    ]


TrendSpec = Literal["linear", "quadratic"]
BreakForm = Literal["level", "level+trend"]  # только сдвиг уровня или сдвиг уровня + наклона


@dataclass(frozen=True)
class DummyModelResult:
    name: str
    break_date: pd.Timestamp
    spec: TrendSpec
    form: BreakForm
    model: sm.regression.linear_model.RegressionResultsWrapper
    fitted_df: pd.DataFrame
    compare_row: dict[str, object]
    diagnostics_row: dict[str, object]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values("date").reset_index(drop=True)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["date", "value"]).reset_index(drop=True)

    d["t"] = np.arange(1, len(d) + 1, dtype=int)
    d["t2"] = d["t"] ** 2
    return d


def _build_X(d: pd.DataFrame, spec: TrendSpec, form: BreakForm, break_date: pd.Timestamp) -> pd.DataFrame:
    D = (d["date"] >= break_date).astype(int)
    d = d.copy()
    d["D"] = D
    d["tD"] = d["t"] * d["D"]

    if spec == "linear":
        base_cols = ["t"]
    elif spec == "quadratic":
        base_cols = ["t", "t2"]
    else:
        raise ValueError(f"Unknown spec: {spec}")

    if form == "level":
        cols = base_cols + ["D"]
    elif form == "level+trend":
        cols = base_cols + ["D", "tD"]
    else:
        raise ValueError(f"Unknown form: {form}")

    X = sm.add_constant(d[cols], has_constant="add")
    return X


def _fit(y: pd.Series, X: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, X).fit()


def _compare_metrics(res: sm.regression.linear_model.RegressionResultsWrapper, name: str) -> dict[str, object]:
    resid = res.resid.astype(float)
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    return {
        "model": name,
        "n": int(res.nobs),
        "k": int(res.df_model) + 1,
        "R2": float(res.rsquared),
        "Adj_R2": float(res.rsquared_adj),
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "RMSE": rmse,
        "MAE": mae,
    }


def _diagnostics(res: sm.regression.linear_model.RegressionResultsWrapper, lb_lags: int = 12) -> dict[str, object]:
    resid = res.resid.astype(float)
    X = res.model.exog

    dw = float(sm.stats.stattools.durbin_watson(resid))

    lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_p = float(lb["lb_pvalue"].iloc[0])

    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
    jb_stat, jb_p, _, _ = jarque_bera(resid)

    return {
        "DW": dw,
        f"LB({lb_lags})_stat": lb_stat,
        f"LB({lb_lags})_p": lb_p,
        "BP_stat": float(bp_stat),
        "BP_p": float(bp_p),
        "JB_stat": float(jb_stat),
        "JB_p": float(jb_p),
    }


def fit_dummy_models(
    df: pd.DataFrame,
    break_date: str | pd.Timestamp,
    *,
    spec: TrendSpec = "linear",
    lb_lags: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, DummyModelResult]]:
    """
    Строит 3 модели:
    - baseline (без dummy)
    - level shift: +D
    - level+trend shift: +D + t*D

    Возвращает:
    - compare_tbl (сравнение по R2/AIC/BIC/RMSE/MAE)
    - diag_tbl (DW, Ljung-Box, BP, JB)
    - models (словарь с результатами и fitted_df для графиков)
    """
    d = _prepare(df)
    bd = pd.Timestamp(break_date)

    y = d["value"]

    # baseline
    X0 = _build_X(d, spec=spec, form="level", break_date=pd.Timestamp("2100-01-01"))  # D=0 всегда
    # в X0 сейчас есть "D", но он нулевой; проще удалить столбец D чтобы было честно
    X0 = X0.drop(columns=[c for c in X0.columns if c in {"D", "tD"}], errors="ignore")
    res0 = _fit(y, X0)
    name0 = f"Baseline ({spec} trend)"

    # level
    X1 = _build_X(d, spec=spec, form="level", break_date=bd)
    res1 = _fit(y, X1)
    name1 = f"Dummy level shift @ {bd.date()} ({spec})"

    # level + trend
    X2 = _build_X(d, spec=spec, form="level+trend", break_date=bd)
    res2 = _fit(y, X2)
    name2 = f"Dummy level+trend shift @ {bd.date()} ({spec})"

    results = [(name0, bd, spec, "level", res0), (name1, bd, spec, "level", res1), (name2, bd, spec, "level+trend", res2)]

    models: dict[str, DummyModelResult] = {}
    compare_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    for name, bd_i, spec_i, form_i, res in results:
        compare = _compare_metrics(res, name=name)
        diag = {"model": name, **_diagnostics(res, lb_lags=lb_lags)}

        fitted = d[["date", "value"]].copy()
        fitted["y_hat"] = res.fittedvalues
        fitted["resid"] = res.resid

        models[name] = DummyModelResult(
            name=name,
            break_date=bd_i,
            spec=spec_i,
            form=form_i,  # baseline формально "level", но без D
            model=res,
            fitted_df=fitted,
            compare_row=compare,
            diagnostics_row=diag,
        )
        compare_rows.append(compare)
        diag_rows.append(diag)

    compare_tbl = pd.DataFrame(compare_rows).sort_values(["AIC", "BIC"], ascending=[True, True]).reset_index(drop=True)
    diag_tbl = pd.DataFrame(diag_rows).reset_index(drop=True)

    return compare_tbl, diag_tbl, models


def extract_key_coefs(res: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """
    Удобная табличка коэффициентов для вставки в отчёт.
    """
    params = res.params
    se = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    out = pd.DataFrame({"coef": params, "std_err": se, "t": tvals, "p_value": pvals})
    return out


def save_step3_plots(models: dict[str, DummyModelResult], out_dir: str = "plots_step3") -> None:
    import os

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    for name, mr in models.items():
        dfp = mr.fitted_df

        # факт vs прогноз
        plt.figure()
        plt.plot(dfp["date"], dfp["value"], label="actual")
        plt.plot(dfp["date"], dfp["y_hat"], label="fitted")
        plt.title(name)
        plt.ylabel("%")
        plt.xlabel("date")
        plt.legend()
        plt.tight_layout()
        safe = name.replace(" ", "_").replace(":", "").replace("@", "").replace("/", "_")
        plt.savefig(f"{out_dir}/fit_{safe}.png", dpi=200)
        plt.close()

        # остатки
        plt.figure()
        plt.plot(dfp["date"], dfp["resid"])
        plt.axhline(0.0, linewidth=1)
        plt.title(f"Residuals: {name}")
        plt.ylabel("residual")
        plt.xlabel("date")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/resid_{safe}.png", dpi=200)
        plt.close()
