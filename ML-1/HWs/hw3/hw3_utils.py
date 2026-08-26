from __future__ import annotations

import ast
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def gini(y_true, y_score) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1.0


def load_optional_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    if "duration" in result:
        median_duration = result["duration"].median()
        result["duration_missing"] = result["duration"].isna().astype(int)
        result["duration"] = result["duration"].fillna(median_duration)
        result["log_duration"] = np.log1p(result["duration"])
        result["sqrt_duration"] = np.sqrt(result["duration"].clip(lower=0))
        result["is_long_match"] = (result["duration"] > median_duration).astype(int)
        result["duration_bin"] = pd.qcut(
            result["duration"],
            q=5,
            labels=False,
            duplicates="drop",
        ).astype(float)

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["weekday"] = result["date"].dt.weekday
    result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)
    result["season"] = ((result["month"] - 1) // 3 + 1).astype(float)
    result["mode_region"] = result["game_mode"].astype(str) + "_" + result["region"].astype(str)
    result["mode_region_freq"] = result["mode_region"].map(result["mode_region"].value_counts())
    return result


def split_feature_columns(df: pd.DataFrame, target: str = "radiant_win") -> tuple[list[str], list[str]]:
    drop_cols = {target, "match_id", "date"}
    feature_cols = [col for col in df.columns if col not in drop_cols]
    categorical = [
        col
        for col in feature_cols
        if df[col].dtype == "object" or col in {"region", "game_mode", "weekday", "month", "season", "mode_region"}
    ]
    numeric = [col for col in feature_cols if col not in categorical]
    return numeric, categorical


def make_logreg_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_pipeline(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    start = time.time()
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]
    return {
        "train_gini": gini(y_train, train_pred),
        "val_gini": gini(y_val, val_pred),
        "fit_seconds": time.time() - start,
    }


def tune_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, Pipeline]:
    grid = ParameterGrid(
        {
            "C": [0.05, 0.2, 1.0, 3.0],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        }
    )
    rows = []
    best_score = -np.inf
    best_model = None

    for params in grid:
        model = make_logreg_pipeline(numeric_features, categorical_features, **params)
        metrics = evaluate_pipeline(model, X_train, y_train, X_val, y_val)
        row = {**params, **metrics}
        rows.append(row)
        if metrics["val_gini"] > best_score:
            best_score = metrics["val_gini"]
            best_model = model

    return pd.DataFrame(rows).sort_values("val_gini", ascending=False), best_model


def feature_selection_grid(
    fitted_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    Xt_train = fitted_model.named_steps["preprocess"].transform(X_train)
    Xt_val = fitted_model.named_steps["preprocess"].transform(X_val)
    base = LogisticRegression(C=0.2, penalty="l1", solver="liblinear", max_iter=1000, random_state=42)
    base.fit(Xt_train, y_train)

    rows = []
    for pct in [0.25, 0.5, 0.75, 1.0]:
        threshold = np.quantile(np.abs(base.coef_).ravel(), 1 - pct)
        selector = SelectFromModel(base, threshold=threshold, prefit=True)
        Xs_train = selector.transform(Xt_train)
        Xs_val = selector.transform(Xt_val)
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(Xs_train, y_train)
        pred = model.predict_proba(Xs_val)[:, 1]
        rows.append(
            {
                "feature_share": pct,
                "n_features": Xs_train.shape[1],
                "val_gini": gini(y_val, pred),
                "score_with_penalty": gini(y_val, pred) - 0.01 * (1 - pct),
            }
        )

    return pd.DataFrame(rows).sort_values("score_with_penalty", ascending=False)


class HeroesEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, n_heroes: int = 1):
        self.n_heroes = n_heroes
        self.hero_to_col_: dict[int, int] | None = None
        self.columns_: list[str] | None = None

    def fit(self, X, y=None):
        heroes = pd.Index(X["hero_id"].dropna().astype(int).unique()).sort_values()
        self.hero_to_col_ = {hero_id: idx for idx, hero_id in enumerate(heroes)}
        self.columns_ = [f"hero_{hero_id}" for hero_id in heroes]
        return self

    def transform(self, X, y=None):
        if self.hero_to_col_ is None or self.columns_ is None:
            raise RuntimeError("Call fit before transform.")

        match_ids = pd.Index(X["match_id"].dropna().unique())
        match_to_row = {match_id: idx for idx, match_id in enumerate(match_ids)}
        matrix = np.zeros((len(match_ids), len(self.hero_to_col_)), dtype=np.int8)

        for row in X[["match_id", "hero_id", "player_slot"]].dropna().itertuples(index=False):
            hero_id = int(row.hero_id)
            if hero_id not in self.hero_to_col_:
                continue
            sign = 1 if int(row.player_slot) < 128 else -1
            matrix[match_to_row[row.match_id], self.hero_to_col_[hero_id]] += sign

        return pd.DataFrame(matrix, index=match_ids, columns=self.columns_)


class WinrateEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, n_heroes: int = 1):
        self.n_heroes = n_heroes
        self.hero_winrate_: pd.Series | None = None

    def fit(self, player_df: pd.DataFrame, matches_df: pd.DataFrame):
        joined = player_df.merge(matches_df[["match_id", "radiant_win"]], on="match_id", how="inner")
        joined["is_radiant"] = joined["player_slot"].astype(int) < 128
        joined["hero_won"] = joined["is_radiant"].eq(joined["radiant_win"])
        self.hero_winrate_ = joined.groupby("hero_id")["hero_won"].mean()
        return self

    def transform(self, player_df: pd.DataFrame, y=None):
        if self.hero_winrate_ is None:
            raise RuntimeError("Call fit before transform.")

        temp = player_df.copy()
        temp["is_radiant"] = temp["player_slot"].astype(int) < 128
        temp["hero_winrate"] = temp["hero_id"].map(self.hero_winrate_).fillna(self.hero_winrate_.mean())
        temp["signed_winrate"] = np.where(temp["is_radiant"], temp["hero_winrate"], -temp["hero_winrate"])
        return temp.groupby("match_id")["signed_winrate"].agg(
            radiant_minus_dire_winrate="sum",
            avg_abs_hero_winrate=lambda s: np.mean(np.abs(s)),
        )


def parse_adv_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        return np.array([], dtype=float)
    try:
        return np.asarray(ast.literal_eval(str(value)), dtype=float)
    except (ValueError, SyntaxError):
        return np.array([], dtype=float)


class TrendExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, agg_func: Callable[[np.ndarray], float] | None = None):
        self.agg_func = agg_func or (lambda arr: arr[-1] - arr[0] if len(arr) else np.nan)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = pd.DataFrame(index=X.index)
        for col in [col for col in X.columns if col.endswith("_adv")]:
            result[f"{col}_trend"] = X[col].map(lambda value: self.agg_func(parse_adv_array(value)))
        return result


class AUCExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, agg_func: Callable[[np.ndarray], float] | None = None):
        self.agg_func = agg_func or (lambda arr: np.trapz(arr) if len(arr) else np.nan)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = pd.DataFrame(index=X.index)
        for col in [col for col in X.columns if col.endswith("_adv")]:
            result[f"{col}_auc"] = X[col].map(lambda value: self.agg_func(parse_adv_array(value)))
        return result


def pairwise_mode_region_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["mode_region"] = result["game_mode"].astype(str) + "_" + result["region"].astype(str)
    result["mode_region_freq"] = result["mode_region"].map(result["mode_region"].value_counts())
    return result
