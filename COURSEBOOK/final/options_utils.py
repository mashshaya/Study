import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
import statsmodels.formula.api as smf


def black76_price(F, K, T, r, sigma, option_type):
    F, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, r, sigma))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = np.exp(-r * T)
    call = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(np.asarray(option_type) == "call", call, put)


def american_futures_option_crr(F, K, T, r, sigma, option_type, steps=200):
    if not all(np.isfinite(v) for v in (F, K, T, r, sigma)):
        return np.nan
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    if option_type not in {"call", "put"}:
        return np.nan
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    if not (np.isfinite(u) and np.isfinite(d) and u != d):
        return np.nan
    p = (1.0 - d) / (u - d)
    if not (0 <= p <= 1):
        return np.nan
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    F_mat = F * (u**j) * (d ** (steps - j))
    values = np.maximum(F_mat - K, 0.0) if option_type == "call" else np.maximum(K - F_mat, 0.0)
    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:step + 2] + (1 - p) * values[:step + 1])
        j = np.arange(step + 1)
        F_now = F * (u**j) * (d ** (step - j))
        exercise = np.maximum(F_now - K, 0.0) if option_type == "call" else np.maximum(K - F_now, 0.0)
        values = np.maximum(values, exercise)
    return float(values[0])


def compute_expanding_garch_forecast(underlying_frame, min_obs=63, refit_every=21):
    rows = []
    for secid, part in underlying_frame.groupby("UNDERLYING_SECID", observed=False):
        part = part.sort_values("TRADEDATE").copy()
        returns = part["ret_1d"].to_numpy()
        forecasts = [np.nan] * len(part)
        fit = params = None
        last_variance = np.nan
        for i in range(min_obs, len(part)):
            hist = returns[:i]
            hist = hist[np.isfinite(hist)]
            if len(hist) < min_obs:
                continue
            if fit is None or (i - min_obs) % refit_every == 0:
                try:
                    model = arch_model(hist * 100.0, mean="Zero", vol="GARCH", p=1, q=1,
                                       dist="normal", rescale=False)
                    fit = model.fit(disp="off", update_freq=0)
                    params = fit.params
                    last_variance = np.nan
                except Exception:
                    fit = params = None
                    last_variance = np.nan
            if fit is None or params is None:
                continue
            try:
                if np.isfinite(last_variance) and np.isfinite(returns[i - 1]):
                    forecast_var = (
                        float(params["omega"])
                        + float(params["alpha[1]"]) * float((returns[i - 1] * 100.0) ** 2)
                        + float(params["beta[1]"]) * float(last_variance)
                    )
                else:
                    forecast_var = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
                last_variance = forecast_var
                forecasts[i] = np.sqrt(forecast_var) / 100.0 * np.sqrt(252)
            except Exception:
                forecasts[i] = np.nan
        out = part[["TRADEDATE", "UNDERLYING_SECID"]].copy()
        out["garch_vol"] = forecasts
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["TRADEDATE", "UNDERLYING_SECID", "garch_vol"])
    return pd.concat(rows, ignore_index=True)


def add_model_block(frame, sigma_col, model_prefix, use_american=True, steps=100):
    out = frame.copy()
    valid = (
        (out["market_price"] > 0) & (out["F"] > 0) & (out["K"] > 0)
        & (out["T"] > 0) & (out[sigma_col] > 0)
        & out["option_type_norm"].isin(["call", "put"])
    )
    price_col = f"model_price_{model_prefix}"
    out[price_col] = np.nan
    if use_american:
        out.loc[valid, price_col] = out.loc[valid].apply(
            lambda r: american_futures_option_crr(
                float(r["F"]), float(r["K"]), float(r["T"]), float(r["r"]),
                float(r[sigma_col]), str(r["option_type_norm"]), steps,
            ), axis=1,
        )
    else:
        out.loc[valid, price_col] = black76_price(
            out.loc[valid, "F"], out.loc[valid, "K"], out.loc[valid, "T"],
            out.loc[valid, "r"], out.loc[valid, sigma_col], out.loc[valid, "option_type_norm"],
        )
    err = out[price_col] - out["market_price"]
    out[f"error_{model_prefix}"] = err
    out[f"abs_error_{model_prefix}"] = err.abs()
    out[f"pct_error_{model_prefix}"] = err / out["market_price"]
    out[f"squared_error_{model_prefix}"] = err**2
    return out


def restrict_to_common_sample(frame, model_specs):
    mask = np.ones(len(frame), dtype=bool)
    for spec in model_specs.values():
        mask &= frame[spec["price"]].notna().to_numpy()
    return frame.loc[mask].copy()


def calc_metrics(frame, spec):
    sub = frame.loc[frame[spec["price"]].notna()]
    if sub.empty:
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan, "mean_error": np.nan,
                "median_abs_error": np.nan, "mean_abs_pct_error": np.nan}
    return {
        "N": int(len(sub)),
        "MAE": float(sub[spec["abs"]].mean()),
        "RMSE": float(np.sqrt(sub[spec["sq"]].mean())),
        "mean_error": float(sub[spec["err"]].mean()),
        "median_abs_error": float(sub[spec["abs"]].median()),
        "mean_abs_pct_error": float(sub[spec["pct"]].abs().mean()),
    }


def calc_metrics_by_group(frame, spec, group_col):
    sub = frame.loc[frame[spec["price"]].notna()]
    grouped = (
        sub.groupby(group_col, dropna=False, observed=False)
        .agg(N=(spec["err"], "size"), MAE=(spec["abs"], "mean"),
             mean_error=(spec["err"], "mean"), median_abs_error=(spec["abs"], "median"))
        .reset_index()
    )
    rmse = (sub.groupby(group_col, dropna=False, observed=False)[spec["sq"]]
            .apply(lambda s: float(np.sqrt(np.mean(s)))).reset_index(name="RMSE"))
    mape = (sub.groupby(group_col, dropna=False, observed=False)[spec["pct"]]
            .apply(lambda s: float(np.mean(np.abs(s)))).reset_index(name="mean_abs_pct_error"))
    return grouped.merge(rmse, on=group_col, how="left").merge(mape, on=group_col, how="left")


def compute_tvnae_series(frame, spec, tv_threshold=10.0):
    df = frame.loc[frame[spec["price"]].notna()].copy()
    df["intrinsic"] = np.where(
        df["option_type_norm"] == "call",
        np.maximum(df["F"] - df["K"], 0.0),
        np.maximum(df["K"] - df["F"], 0.0),
    )
    df["time_value"] = df["market_price"] - df["intrinsic"]
    df["tvnae"] = df[spec["abs"]] / df["time_value"]
    return df.loc[df["time_value"] >= tv_threshold, ["tvnae", "time_value", "vol_regime", "DTE_bucket"]]


def calc_tvnae_metrics(frame, spec, group_col=None, tv_threshold=10.0):
    tv_df = compute_tvnae_series(frame, spec, tv_threshold)
    if group_col is None:
        return {"N_filtered": len(tv_df), "mean_TVNAE": float(tv_df["tvnae"].mean()),
                "median_TVNAE": float(tv_df["tvnae"].median())}
    return (tv_df.groupby(group_col, observed=False)["tvnae"]
            .agg(N="count", mean_TVNAE="mean", median_TVNAE="median").reset_index())


def mean_error_test(frame, spec, model_name):
    sub = frame[[spec["err"], "TRADEDATE"]].dropna().copy()
    sub["cluster"] = sub["TRADEDATE"].astype(str)
    fit = smf.ols(f'{spec["err"]} ~ 1', data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["cluster"]}
    )
    return {
        "model": model_name, "N": int(len(sub)),
        "mean_error": float(fit.params["Intercept"]),
        "std_error_clustered": float(fit.bse["Intercept"]),
        "t_stat": float(fit.tvalues["Intercept"]),
        "p_value": float(fit.pvalues["Intercept"]),
    }


def dm_test_daily(frame, spec_a, spec_b, model_a, model_b, loss="sq"):
    daily = (frame.groupby("TRADEDATE")[[spec_a[loss], spec_b[loss]]]
             .mean().dropna().reset_index())
    daily["loss_diff"] = daily[spec_a[loss]] - daily[spec_b[loss]]
    fit = smf.ols("loss_diff ~ 1", data=daily).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "comparison": f"{model_a} minus {model_b}",
        "loss_metric": "MSE" if loss == "sq" else "MAE",
        "N_trade_dates": int(len(daily)),
        "mean_loss_diff": float(fit.params["Intercept"]),
        "std_error_hac": float(fit.bse["Intercept"]),
        "t_stat": float(fit.tvalues["Intercept"]),
        "p_value": float(fit.pvalues["Intercept"]),
    }


def price_with_alt_rate(frame, sigma_col, rate_value, steps=100):
    valid = (
        (frame["market_price"] > 0) & (frame["F"] > 0) & (frame["K"] > 0)
        & (frame["T"] > 0) & (frame[sigma_col] > 0)
        & frame["option_type_norm"].isin(["call", "put"])
    )
    prices = pd.Series(np.nan, index=frame.index)
    prices.loc[valid] = frame.loc[valid].apply(
        lambda r: american_futures_option_crr(
            float(r["F"]), float(r["K"]), float(r["T"]), rate_value,
            float(r[sigma_col]), str(r["option_type_norm"]), steps,
        ), axis=1,
    )
    out = pd.DataFrame({"model_price": prices})
    out["error"] = prices - frame["market_price"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2
    out["pct_error"] = out["error"] / frame["market_price"]
    return out
