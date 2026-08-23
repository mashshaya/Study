import textwrap
from pathlib import Path
import nbformat as nbf

ROOT = Path("/Users/maria/Desktop/Code/HSE/COURSEBOOK")
OUTPUT = ROOT / "clear" / "final_notebook.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [

    md("# Оценка опционов MM на фьючерс MXI"),

    code("""
        import sys
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import statsmodels.formula.api as smf

        project_root = Path("/Users/maria/Desktop/Code/HSE/COURSEBOOK")
        sys.path.insert(0, str(project_root / "clear"))
        from options_utils import (
            black76_price, american_futures_option_crr,
            compute_expanding_garch_forecast, add_model_block,
            restrict_to_common_sample, calc_metrics, calc_metrics_by_group,
            compute_tvnae_series, calc_tvnae_metrics,
            mean_error_test, dm_test_daily, price_with_alt_rate,
        )

        results_dir = project_root / "clear" / "results_models"
        figures_dir = results_dir / "figures_final"
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        pd.set_option("display.max_columns", 200)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        plt.style.use("seaborn-v0_8-whitegrid")
    """),

    # ── EDA ──────────────────────────────────────────────────────────────────

    md("## EDA"),

    md("### Загрузка датасета"),
    code("""
        data_path = project_root / "clear" / "mm_options_with_mxi_underlying_2024_2026.parquet"
        eda_df = pd.read_parquet(data_path)
        eda_df["TRADEDATE"] = pd.to_datetime(eda_df["TRADEDATE"], errors="coerce")
        eda_df["expiry_date"] = pd.to_datetime(eda_df["expiry_date"], errors="coerce")
        eda_df.shape
    """),

    md("### Структура и покрытие"),
    code("""
        pd.Series({
            "rows": len(eda_df),
            "unique_options": int(eda_df["SECID"].nunique()),
            "unique_underlyings": int(eda_df["UNDERLYING_SECID"].dropna().nunique()),
            "date_min": eda_df["TRADEDATE"].min().date(),
            "date_max": eda_df["TRADEDATE"].max().date(),
            "expiry_min": eda_df["expiry_date"].min().date(),
            "expiry_max": eda_df["expiry_date"].max().date(),
            "option_types": eda_df["option_type"].value_counts(dropna=False).to_dict(),
        })
    """),

    md("### Контракты underlying и строки по годам"),
    code("""
        display(eda_df["UNDERLYING_SHORTNAME"].value_counts().to_frame("rows"))
        display(eda_df.groupby(eda_df["TRADEDATE"].dt.year).size().to_frame("rows"))
    """),

    md("### Пропуски в ключевых полях"),
    code("""
        eda_df[["market_price", "underlying_price", "moneyness",
                "UNDERLYING_CLOSE", "UNDERLYING_VOLUME"]].isna().mean().sort_values(ascending=False).to_frame("missing_share")
    """),

    # ── ПОДГОТОВКА ВЫБОРКИ ────────────────────────────────────────────────────

    md("## Подготовка выборки"),

    md("### Загрузка и типизация"),
    code("""
        df = pd.read_parquet(data_path)
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce").dt.normalize()
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.normalize()

        for col in ["strike", "SETTLEPRICE", "market_price", "underlying_price",
                    "UNDERLYING_SETTLEPRICE", "UNDERLYING_OPEN", "UNDERLYING_HIGH",
                    "UNDERLYING_LOW", "UNDERLYING_CLOSE", "UNDERLYING_VALUE",
                    "UNDERLYING_VOLUME", "UNDERLYING_OPENPOSITION", "UNDERLYING_NUMTRADES",
                    "DTE_DAYS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["market_price"] = df["SETTLEPRICE"]
        df["F"] = df["underlying_price"]
        df["K"] = df["strike"]
        df["DTE_DAYS"] = (df["expiry_date"] - df["TRADEDATE"]).dt.days
        df["T"] = df["DTE_DAYS"] / 365.0
        df["r"] = 0.16
        df["option_type_norm"] = df["option_type"].astype(str).str.upper().map({"C": "call", "P": "put"})
        df["moneyness"] = df["F"] / df["K"]
        df["log_moneyness"] = np.log(df["moneyness"])
        df.shape
    """),

    md("### Операционный фильтр"),
    code("""
        pd.Series({
            "bad_trade_date":      int(df["TRADEDATE"].isna().sum()),
            "bad_expiry_date":     int(df["expiry_date"].isna().sum()),
            "bad_market_price":    int((df["market_price"] <= 0).sum()),
            "bad_futures_price":   int((df["F"] <= 0).sum()),
            "bad_strike":          int((df["K"] <= 0).sum()),
            "bad_dte":             int((df["DTE_DAYS"] <= 0).sum()),
            "bad_option_type":     int((~df["option_type_norm"].isin(["call", "put"])).sum()),
            "bad_underlying_secid":int(df["UNDERLYING_SECID"].isna().sum()),
        }).sort_values(ascending=False)
    """),
    code("""
        mask = (
            df["TRADEDATE"].notna() & df["expiry_date"].notna()
            & (df["market_price"] > 0) & (df["F"] > 0) & (df["K"] > 0)
            & (df["DTE_DAYS"] > 0) & df["option_type_norm"].isin(["call", "put"])
            & df["UNDERLYING_SECID"].notna()
        )
        modelling_df = (df.loc[mask].copy()
                        .sort_values(["TRADEDATE", "UNDERLYING_SECID", "expiry_date", "K", "option_type_norm"])
                        .reset_index(drop=True))
        print(f"Original: {len(df)}, Modelling: {len(modelling_df)}, Share: {len(modelling_df)/len(df):.4f}")
    """),

    md("### Диагностика хвостов"),
    code("""
        pd.Series({
            "share_moneyness_lt_0.7":           float((modelling_df["moneyness"] < 0.7).mean()),
            "share_moneyness_gt_1.8":           float((modelling_df["moneyness"] > 1.8).mean()),
            "share_dte_gt_365":                 float((modelling_df["DTE_DAYS"] > 365).mean()),
            "share_market_price_gt_K_plus_F":   float((modelling_df["market_price"] > (modelling_df["K"] + modelling_df["F"])).mean()),
        })
    """),

    # ── ВОЛАТИЛЬНОСТЬ И РЕЖИМЫ ────────────────────────────────────────────────

    md("## Волатильность и режимы"),

    md("### Исторические волатильности и режимы"),
    code("""
        underlying_daily = (
            modelling_df[["TRADEDATE", "UNDERLYING_SECID", "UNDERLYING_SHORTNAME", "F"]]
            .drop_duplicates().sort_values(["UNDERLYING_SECID", "TRADEDATE"]).reset_index(drop=True)
        )
        underlying_daily["ret_1d"] = underlying_daily.groupby("UNDERLYING_SECID")["F"].transform(
            lambda s: np.log(s).diff()
        )
        underlying_daily["ret_21d"] = underlying_daily.groupby("UNDERLYING_SECID")["F"].transform(
            lambda s: np.log(s / s.shift(21))
        )
        underlying_daily["hv_21d"] = underlying_daily.groupby("UNDERLYING_SECID")["ret_1d"].transform(
            lambda s: s.rolling(21).std() * np.sqrt(252)
        )
        underlying_daily["hv_63d"] = underlying_daily.groupby("UNDERLYING_SECID")["ret_1d"].transform(
            lambda s: s.rolling(63).std() * np.sqrt(252)
        )
        vol_q = underlying_daily["hv_21d"].dropna().quantile([1/3, 2/3]).to_dict()
        underlying_daily["vol_regime"] = pd.cut(
            underlying_daily["hv_21d"],
            bins=[-np.inf, vol_q[1/3], vol_q[2/3], np.inf],
            labels=["low_vol", "mid_vol", "high_vol"],
        )
        underlying_daily["trend_regime"] = np.select(
            [underlying_daily["ret_21d"] > 0, underlying_daily["ret_21d"] <= 0],
            ["up", "down"], default=pd.NA,
        )
        underlying_daily["regime_label"] = np.where(
            underlying_daily["vol_regime"].isna(), pd.NA,
            underlying_daily["vol_regime"].astype(str) + "_" + underlying_daily["trend_regime"],
        )
        modelling_df = modelling_df.merge(
            underlying_daily[["TRADEDATE", "UNDERLYING_SECID", "ret_1d", "ret_21d",
                               "hv_21d", "hv_63d", "vol_regime", "trend_regime", "regime_label"]],
            on=["TRADEDATE", "UNDERLYING_SECID"], how="left",
        )
        modelling_df[["TRADEDATE", "UNDERLYING_SECID", "F", "hv_21d", "hv_63d", "vol_regime"]].head()
    """),

    # ── GARCH ─────────────────────────────────────────────────────────────────

    md("## GARCH"),

    md("### Expanding forecast (без look-ahead bias)"),
    code("""
        garch_forecasts = compute_expanding_garch_forecast(underlying_daily, min_obs=63, refit_every=21)
        modelling_df = modelling_df.merge(garch_forecasts, on=["TRADEDATE", "UNDERLYING_SECID"], how="left")
        print("garch non-null share:", round(float(modelling_df["garch_vol"].notna().mean()), 4))
    """),

    # ── ЦЕНОВЫЕ МОДЕЛИ ────────────────────────────────────────────────────────

    md("## Ценовые модели"),

    md("### Расчёт модельных цен (CRR и Black-76)"),
    code("""
        pricing_df = modelling_df.copy()
        pricing_df = add_model_block(pricing_df, "hv_21d",   "crr_hv_21d",      use_american=True,  steps=100)
        pricing_df = add_model_block(pricing_df, "hv_63d",   "crr_hv_63d",      use_american=True,  steps=100)
        pricing_df = add_model_block(pricing_df, "garch_vol","crr_garch",        use_american=True,  steps=100)
        pricing_df = add_model_block(pricing_df, "hv_63d",   "black76_hv_63d",  use_american=False)
    """),

    md("### Сегментация выборки"),
    code("""
        pricing_df["moneyness_bucket"] = pd.cut(
            pricing_df["moneyness"],
            bins=[0, 0.9, 0.97, 1.03, 1.1, np.inf],
            labels=["deep_OTM", "OTM", "ATM", "ITM", "deep_ITM"],
        )
        pricing_df["DTE_bucket"] = pd.cut(
            pricing_df["DTE_DAYS"],
            bins=[0, 7, 30, 90, np.inf],
            labels=["0-7", "8-30", "31-90", "91+"],
        )
        pricing_df["abs_log_moneyness"] = pricing_df["log_moneyness"].abs()
        pricing_df[["SECID", "TRADEDATE", "market_price", "F", "K",
                    "model_price_crr_hv_21d", "model_price_crr_hv_63d",
                    "model_price_crr_garch", "model_price_black76_hv_63d"]].head()
    """),

    # ── СРАВНЕНИЕ МОДЕЛЕЙ ─────────────────────────────────────────────────────

    md("## Сравнение моделей"),

    md("### Наборы моделей и общая выборка"),
    code("""
        american_models = {
            "CRR + hv_21d": {
                "price": "model_price_crr_hv_21d", "err": "error_crr_hv_21d",
                "abs": "abs_error_crr_hv_21d", "sq": "squared_error_crr_hv_21d",
                "pct": "pct_error_crr_hv_21d",
            },
            "CRR + hv_63d": {
                "price": "model_price_crr_hv_63d", "err": "error_crr_hv_63d",
                "abs": "abs_error_crr_hv_63d", "sq": "squared_error_crr_hv_63d",
                "pct": "pct_error_crr_hv_63d",
            },
            "CRR + garch": {
                "price": "model_price_crr_garch", "err": "error_crr_garch",
                "abs": "abs_error_crr_garch", "sq": "squared_error_crr_garch",
                "pct": "pct_error_crr_garch",
            },
        }
        euro_benchmark = {
            "Black-76 + hv_63d": {
                "price": "model_price_black76_hv_63d", "err": "error_black76_hv_63d",
                "abs": "abs_error_black76_hv_63d", "sq": "squared_error_black76_hv_63d",
                "pct": "pct_error_black76_hv_63d",
            }
        }
        american_common = restrict_to_common_sample(pricing_df, american_models)
        all_models_common = restrict_to_common_sample(pricing_df, american_models | euro_benchmark)
        pd.DataFrame({
            "sample": ["american only", "american + black76"],
            "rows": [len(american_common), len(all_models_common)],
            "date_min": [american_common["TRADEDATE"].min(), all_models_common["TRADEDATE"].min()],
            "date_max": [american_common["TRADEDATE"].max(), all_models_common["TRADEDATE"].max()],
        })
    """),

    md("### Общие метрики"),
    code("""
        overall_american = pd.DataFrame([
            {"model": name, **calc_metrics(american_common, spec)}
            for name, spec in american_models.items()
        ]).sort_values("MAE").reset_index(drop=True)
        display(overall_american)
    """),

    # ── TVNAE ─────────────────────────────────────────────────────────────────

    md("## TVNAE"),

    md("### Нормированная ошибка по временно́й стоимости"),
    code("""
        TV_THRESHOLD = 10.0

        tvnae_overall = pd.DataFrame([
            {"model": name, **calc_tvnae_metrics(american_common, spec, tv_threshold=TV_THRESHOLD)}
            for name, spec in american_models.items()
        ]).sort_values("mean_TVNAE").reset_index(drop=True)
        display(tvnae_overall)

        tvnae_by_regime = pd.concat([
            calc_tvnae_metrics(american_common, spec, "vol_regime", TV_THRESHOLD).assign(model=name)
            for name, spec in american_models.items()
        ], ignore_index=True)
        display(tvnae_by_regime.pivot(index="model", columns="vol_regime", values="mean_TVNAE").round(3))

        tvnae_by_dte = pd.concat([
            calc_tvnae_metrics(american_common, spec, "DTE_bucket", TV_THRESHOLD).assign(model=name)
            for name, spec in american_models.items()
        ], ignore_index=True)
        display(tvnae_by_dte.pivot(index="model", columns="DTE_bucket", values="mean_TVNAE").round(3))
    """),

    # ── ОШИБКИ ПО СЕГМЕНТАМ ───────────────────────────────────────────────────

    md("## Ошибки по сегментам"),

    md("### Разбивка по режиму, DTE, moneyness"),
    code("""
        breakdown_tables = {}
        for group_col, title in [
            ("vol_regime",       "По режиму волатильности"),
            ("DTE_bucket",       "По DTE bucket"),
            ("moneyness_bucket", "По moneyness bucket"),
        ]:
            tables = []
            for name, spec in american_models.items():
                t = calc_metrics_by_group(american_common, spec, group_col)
                t["model"] = name
                tables.append(t)
            breakdown_tables[group_col] = pd.concat(tables, ignore_index=True)
            print(title)
            display(breakdown_tables[group_col])
    """),

    # ── БЕНЧМАРК BLACK-76 ─────────────────────────────────────────────────────

    md("## Бенчмарк Black-76"),

    md("### European benchmark и early exercise premium"),
    code("""
        benchmark_compare = pd.DataFrame([
            {"model": name, **calc_metrics(all_models_common, spec)}
            for name, spec in (american_models | euro_benchmark).items()
        ]).sort_values("MAE").reset_index(drop=True)
        display(benchmark_compare)

        all_models_common["early_exercise_premium"] = (
            all_models_common["model_price_crr_hv_63d"] - all_models_common["model_price_black76_hv_63d"]
        )
        premium_summary = (
            all_models_common.groupby(["vol_regime", "DTE_bucket"], dropna=False, observed=False)
            ["early_exercise_premium"].agg(["count", "mean", "median"]).reset_index()
        )
        display(premium_summary)

        black_below_intrinsic = np.where(
            all_models_common["option_type_norm"].eq("call"),
            all_models_common["model_price_black76_hv_63d"] < np.maximum(all_models_common["F"] - all_models_common["K"], 0),
            all_models_common["model_price_black76_hv_63d"] < np.maximum(all_models_common["K"] - all_models_common["F"], 0),
        )
        pd.DataFrame({
            "metric": ["N common sample", "share Black-76 below intrinsic",
                       "mean early exercise premium", "median early exercise premium"],
            "value": [len(all_models_common), float(np.mean(black_below_intrinsic)),
                      float(all_models_common["early_exercise_premium"].mean()),
                      float(all_models_common["early_exercise_premium"].median())],
        })
    """),

    # ── ЭКОНОМЕТРИЧЕСКИЕ ТЕСТЫ ────────────────────────────────────────────────

    md("## Эконометрические тесты"),

    md("### Тест на нулевую среднюю ошибку"),
    code("""
        mean_error_tests = pd.DataFrame([
            mean_error_test(american_common, spec, name)
            for name, spec in american_models.items()
        ]).sort_values("p_value").reset_index(drop=True)
        display(mean_error_tests)
    """),

    md("### Pairwise DM-тесты"),
    code("""
        model_names = list(american_models.keys())
        pairwise_sq, pairwise_abs = [], []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                pairwise_sq.append(dm_test_daily(american_common, american_models[a], american_models[b], a, b, "sq"))
                pairwise_abs.append(dm_test_daily(american_common, american_models[a], american_models[b], a, b, "abs"))

        dm_sq_table = pd.DataFrame(pairwise_sq).sort_values("comparison").reset_index(drop=True)
        dm_abs_table = pd.DataFrame(pairwise_abs).sort_values("comparison").reset_index(drop=True)
        print("MSE loss differential")
        display(dm_sq_table)
        print("MAE loss differential")
        display(dm_abs_table)
    """),

    md("### OLS на ошибках (3 спецификации)"),
    code("""
        ols_df = american_common[["TRADEDATE", "abs_error_crr_hv_63d", "vol_regime",
                                  "DTE_DAYS", "log_moneyness", "option_type_norm"]].dropna().copy()
        ols_df["high_vol_dummy"] = (ols_df["vol_regime"] == "high_vol").astype(int)
        ols_df["abs_log_moneyness"] = ols_df["log_moneyness"].abs()
        ols_df["cluster"] = ols_df["TRADEDATE"].astype(str)

        ols1 = smf.ols("abs_error_crr_hv_63d ~ high_vol_dummy", data=ols_df).fit(
            cov_type="cluster", cov_kwds={"groups": ols_df["cluster"]}
        )
        ols2 = smf.ols("abs_error_crr_hv_63d ~ high_vol_dummy + DTE_DAYS + abs_log_moneyness", data=ols_df).fit(
            cov_type="cluster", cov_kwds={"groups": ols_df["cluster"]}
        )
        ols3 = smf.ols("abs_error_crr_hv_63d ~ high_vol_dummy + DTE_DAYS + abs_log_moneyness + C(option_type_norm)", data=ols_df).fit(
            cov_type="cluster", cov_kwds={"groups": ols_df["cluster"]}
        )
        print(ols1.summary())
        print(ols2.summary())
        print(ols3.summary())
    """),

    # ── ЧУВСТВИТЕЛЬНОСТЬ К СТАВКЕ ─────────────────────────────────────────────

    md("## Чувствительность к ставке"),

    md("### Grid по r"),
    code("""
        rate_rows = []
        for r_val in [0.12, 0.16, 0.20]:
            tmp = price_with_alt_rate(american_common, "hv_63d", r_val, steps=100)
            valid = tmp["model_price"].notna()
            rate_rows.append({
                "r": r_val, "N": int(valid.sum()),
                "MAE": float(tmp.loc[valid, "abs_error"].mean()),
                "RMSE": float(np.sqrt(tmp.loc[valid, "sq_error"].mean())),
                "mean_error": float(tmp.loc[valid, "error"].mean()),
                "mean_abs_pct_error": float(tmp.loc[valid, "pct_error"].abs().mean()),
            })
        pd.DataFrame(rate_rows).sort_values("MAE").reset_index(drop=True)
    """),

    # ── ROBUSTNESS ────────────────────────────────────────────────────────────

    md("## Robustness"),

    md("### Альтернативные подвыборки"),
    code("""
        robustness_samples = {
            "full":              american_common,
            "excl_deep_tails":   american_common.loc[~american_common["moneyness_bucket"].astype(str).isin(["deep_OTM", "deep_ITM"])].copy(),
            "short_dte_le_91":   american_common.loc[american_common["DTE_DAYS"] <= 91].copy(),
            "long_dte_gt_91":    american_common.loc[american_common["DTE_DAYS"] > 91].copy(),
            "non_high_vol":      american_common.loc[american_common["vol_regime"] != "high_vol"].copy(),
        }
        robustness_rows = []
        for sname, sdf in robustness_samples.items():
            for mname, spec in american_models.items():
                robustness_rows.append({"sample": sname, "model": mname, **calc_metrics(sdf, spec)})
        robustness_table = pd.DataFrame(robustness_rows).sort_values(["sample", "MAE"]).reset_index(drop=True)
        display(robustness_table)
    """),

    # ── ГРАФИКИ ───────────────────────────────────────────────────────────────

    md("## Графики"),

    md("### MAE по моделям и сегментам"),
    code("""
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].bar(overall_american["model"], overall_american["MAE"],
                    color=["#4C78A8", "#F58518", "#54A24B"])
        axes[0].set_title("Overall MAE")
        axes[0].set_ylabel("MAE")
        axes[0].tick_params(axis="x", rotation=20)

        vol_plot = (breakdown_tables["vol_regime"].dropna(subset=["vol_regime"])
                    .pivot(index="vol_regime", columns="model", values="MAE"))
        vol_plot.plot(kind="bar", ax=axes[1])
        axes[1].set_title("MAE by vol regime")
        axes[1].tick_params(axis="x", rotation=0)

        dte_plot = (breakdown_tables["DTE_bucket"].dropna(subset=["DTE_bucket"])
                    .pivot(index="DTE_bucket", columns="model", values="MAE"))
        dte_plot.plot(kind="bar", ax=axes[2])
        axes[2].set_title("MAE by DTE bucket")
        axes[2].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        plt.savefig(figures_dir / "fig_01_03_model_comparison.png", dpi=150)
        plt.show()
    """),

    md("### Rolling MAE"),
    code("""
        rolling = (
            american_common.groupby("TRADEDATE")[
                ["abs_error_crr_hv_21d", "abs_error_crr_hv_63d", "abs_error_crr_garch"]
            ].mean().rolling(21).mean().reset_index()
        )
        plt.figure(figsize=(10, 4))
        plt.plot(rolling["TRADEDATE"], rolling["abs_error_crr_hv_21d"], label="CRR + hv_21d")
        plt.plot(rolling["TRADEDATE"], rolling["abs_error_crr_hv_63d"], label="CRR + hv_63d")
        plt.plot(rolling["TRADEDATE"], rolling["abs_error_crr_garch"],  label="CRR + garch")
        plt.title("Rolling 21-day MAE")
        plt.ylabel("MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_04_rolling_mae.png", dpi=150)
        plt.show()
    """),

    md("### Early exercise premium по DTE"),
    code("""
        prem_by_dte = (
            all_models_common.groupby("DTE_bucket", dropna=False, observed=False)["early_exercise_premium"]
            .mean().reset_index()
        )
        plt.figure(figsize=(8, 4))
        plt.bar(prem_by_dte["DTE_bucket"].astype(str), prem_by_dte["early_exercise_premium"],
                color="#E45756")
        plt.title("Early exercise premium by DTE bucket")
        plt.ylabel("CRR(hv_63d) − Black-76(hv_63d)")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_05_early_exercise_premium_by_dte.png", dpi=150)
        plt.show()
    """),

    # ── СОХРАНЕНИЕ ────────────────────────────────────────────────────────────

    md("## Сохранение артефактов"),

    code("""
        modelling_df.to_parquet(results_dir / "modelling_sample_mxi_clean.parquet", index=False)
        pricing_df.to_parquet(results_dir / "pricing_results_mxi_clean.parquet", index=False)

        overall_american.to_csv(results_dir / "overall_american_models.csv", index=False)
        breakdown_tables["vol_regime"].to_csv(results_dir / "breakdown_by_vol_regime.csv", index=False)
        breakdown_tables["DTE_bucket"].to_csv(results_dir / "breakdown_by_dte_bucket.csv", index=False)
        breakdown_tables["moneyness_bucket"].to_csv(results_dir / "breakdown_by_moneyness_bucket.csv", index=False)
        benchmark_compare.to_csv(results_dir / "benchmark_compare_with_black76.csv", index=False)
        premium_summary.to_csv(results_dir / "early_exercise_premium_summary.csv", index=False)
        mean_error_tests.to_csv(results_dir / "mean_error_tests.csv", index=False)
        dm_sq_table.to_csv(results_dir / "pairwise_dm_sq.csv", index=False)
        dm_abs_table.to_csv(results_dir / "pairwise_dm_abs.csv", index=False)
        robustness_table.to_csv(results_dir / "robustness_checks.csv", index=False)
        tvnae_overall.to_csv(results_dir / "tvnae_overall.csv", index=False)
        tvnae_by_regime.to_csv(results_dir / "tvnae_by_vol_regime.csv", index=False)
        tvnae_by_dte.to_csv(results_dir / "tvnae_by_dte.csv", index=False)

        ols_text = ("OLS SPECIFICATION 1\\n" + ols1.summary().as_text()
                    + "\\n\\nOLS SPECIFICATION 2\\n" + ols2.summary().as_text()
                    + "\\n\\nOLS SPECIFICATION 3\\n" + ols3.summary().as_text())
        (results_dir / "error_ols_results.txt").write_text(ols_text)

        print("Saved to:", results_dir)
    """),

]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUTPUT.write_text(nbf.writes(nb))
print(OUTPUT)
