
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from typing import Optional
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt
from collections.abc import Iterable


class Boosting(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = False,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        subsample=1.0,
        bootstrap_type=None,
        bagging_temperature=1.0,
        rsm=1.0,
        dart: bool = False,
        dropout_rate: float = 0.05,

    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)

        self.models = []
        self.gammas = []

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose
        self.classes_ = np.array([-1, 1])  # в нашей задаче классы захардкожены

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y / (1.0 + np.exp(y * z))  # Исправьте формулу на правильную.
        self._train_predictions: Optional[np.ndarray] = None

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.subsample = subsample
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature

        self._rng = np.random.default_rng(random_state)
        self.rsm = rsm
        self.feature_masks = []
        self.dart = dart
        self.dropout_rate = dropout_rate


    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]

        if self._train_predictions is None:
            self._train_predictions = np.zeros(n, dtype=float)

        if self.dart and len(self.models) > 0:
            dropped = self._dart_dropout()

            kept_pred = np.zeros(n)
            for i, (model, gamma, mask) in enumerate(
                zip(self.models, self.gammas, self.feature_masks)
            ):
                if i in dropped:
                    continue

                if mask is None:
                    pred = model.predict(X)
                else:
                    pred = model.predict(X[:, mask])

                kept_pred += self.learning_rate * gamma * pred

            residuals = self.loss_derivative(y, kept_pred)

        else:
            residuals = self.loss_derivative(y, self._train_predictions)

        Xb, resb, sample_weight = self._bootstrap(X, residuals)

        feature_mask = self._sample_features(X.shape[1])
        model = self._base_model()

        if feature_mask is None:
            X_train = Xb
            X_full = X
        else:
            X_train = Xb[:, feature_mask]
            X_full = X[:, feature_mask]

        if sample_weight is None:
            model.fit(X_train, resb)
        else:
            model.fit(X_train, resb, sample_weight=sample_weight)

        new_pred = model.predict(X_full)

        gamma = self.find_optimal_gamma(y, self._train_predictions, new_pred)

        if self.dart and len(self.models) > 0:
            k = len(dropped)

            if k > 0:
                scale_new = 1.0 / (k + 1)
                scale_old = k / (k + 1)

                for i in dropped:
                    self.gammas[i] *= scale_old

                gamma *= scale_new

        self._train_predictions += self.learning_rate * gamma * new_pred

        self.models.append(model)
        self.gammas.append(float(gamma))
        self.feature_masks.append(feature_mask)

        self.history["train_loss"].append(
            self.loss_fn(y, self._train_predictions)
        )
        self.history["train_roc_auc"].append(
            roc_auc_score(y == 1, self.sigmoid(self._train_predictions))
        )

        return self


    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        use_best_model: bool = False,
    ):
        self.models = []
        self.gammas = []
        self.history = defaultdict(list)

        self._train_predictions = np.zeros(X_train.shape[0])

        if eval_set is not None:
            X_valid, y_valid = eval_set
            valid_predictions = np.zeros(X_valid.shape[0])
            valid_predictions_history = []

        best_score = -np.inf
        best_iter = 0
        rounds_without_improve = 0

        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm(iterator)

        for i in iterator:
            self.partial_fit(X_train, y_train)

            if eval_set is not None:
                
                model = self.models[-1]
                gamma = self.gammas[-1]
                mask = self.feature_masks[-1]

                if mask is None:
                    pred = model.predict(X_valid)
                else:
                    pred = model.predict(X_valid[:, mask])

                valid_predictions += self.learning_rate * gamma * pred
                valid_loss = self.loss_fn(y_valid, valid_predictions)
                self.history["valid_loss"].append(valid_loss)

                valid_auc = roc_auc_score(
                    y_valid == 1, self.sigmoid(valid_predictions)
                )
                self.history["valid_roc_auc"].append(valid_auc)

                metric = valid_auc if self.eval_metric is None else self.history[self.eval_metric][-1]

                if metric > best_score:
                    best_score = metric
                    best_iter = i
                    rounds_without_improve = 0
                else:
                    rounds_without_improve += 1

                if (
                    self.early_stopping_rounds
                    and rounds_without_improve >= self.early_stopping_rounds
                ):
                    if self.verbose:
                        print(f"Early stopping at iteration {i}")
                    break

        if use_best_model and eval_set is not None:
            self.models = self.models[: best_iter + 1]
            self.gammas = self.gammas[: best_iter + 1]

        for key in self.history:
            self.history[key] = np.array(self.history[key])

        return self


    def predict_proba(self, X: np.ndarray):
        if len(self.models) == 0:
            p = np.full(X.shape[0], 0.5, dtype=float)
            return np.column_stack([1.0 - p, p])

        z = np.zeros(X.shape[0], dtype=float)

        for model, gamma, mask in zip(self.models, self.gammas, self.feature_masks):
            if mask is None:
                z += self.learning_rate * gamma * model.predict(X)
            else:
                z += self.learning_rate * gamma * model.predict(X[:, mask])

        p = self.sigmoid(z)
        return np.column_stack([1.0 - p, p])


    def predict(self, X: np.ndarray):
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, 1, -1)


    def find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(
                y,
                old_predictions + self.learning_rate * gamma * new_predictions
            )
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]


    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])


    def _base_model(self):
        params = dict(self.base_model_params)

        if self.random_state is not None and "random_state" not in params:
            try:
                self.base_model_class(random_state=0)
                params["random_state"] = self.random_state
            except TypeError:
                pass

        return self.base_model_class(**params)


    def plot_history(self, keys):
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            plt.plot(self.history[key], label=key)

        plt.legend()
        plt.grid(True)
        plt.show()

    def _bootstrap(self, X, y):
        n = X.shape[0]
        rng = self._rng

        if self.bootstrap_type is None:
            return X, y, None

        if self.bootstrap_type == "Bernoulli":
            mask = rng.random(n) < self.subsample
            return X[mask], y[mask], None

        if self.bootstrap_type == "Bayesian":
            u = rng.random(n)
            weights = (-np.log(u)) ** self.bagging_temperature
            return X, y, weights

        raise ValueError(f"Unknown bootstrap_type: {self.bootstrap_type}")
    

    def _sample_features(self, n_features):
        if self.rsm >= 1.0:
            return None

        mask = self._rng.random(n_features) < self.rsm

        if not mask.any():
            mask[self._rng.integers(0, n_features)] = True

        return mask
    

    @property
    def feature_importances_(self) -> np.ndarray:
        if len(self.models) == 0:
            return None

        n_features = self._train_predictions.shape[0]
        n_features = max(
            (
                model.n_features_in_
                if mask is None
                else max(np.where(mask)[0]) + 1
            )
            for model, mask in zip(self.models, self.feature_masks)
        )

        total_importance = np.zeros(n_features, dtype=float)

        for model, gamma, mask in zip(self.models, self.gammas, self.feature_masks):
            imp = model.feature_importances_
            weight = abs(gamma)

            if mask is None:
                total_importance[: len(imp)] += weight * imp
            else:
                full_imp = np.zeros(n_features)
                full_imp[mask] = imp
                total_importance += weight * full_imp

        if total_importance.sum() > 0:
            total_importance /= total_importance.sum()

        return total_importance


    def _dart_dropout(self):
        n_models = len(self.models)
        if n_models == 0:
            return []

        drop_mask = self._rng.random(n_models) < self.dropout_rate
        dropped = np.where(drop_mask)[0].tolist()

        if len(dropped) == n_models:
            dropped = dropped[:-1]

        return dropped
