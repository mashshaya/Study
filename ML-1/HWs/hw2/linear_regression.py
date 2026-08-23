import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional
from scipy.sparse.linalg import svds
import numpy as np


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []


    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w


    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l = X.shape[0]
        e = X @ self.w - y
        
        if self.loss_function is LossFunction.MSE:
            gradient = (2 / l) * (X.T @ e) + 2 * self.l2_coef * self.w
            
        elif self.loss_function is LossFunction.MAE:
            gradient = (1 / l) * (X.T @ np.sign(e))+ 2 * self.l2_coef * self.w

        elif self.loss_function is LossFunction.LogCosh:
            gradient = (1 / l) * (X.T @ np.tanh(e)) + 2 * self.l2_coef * self.w

        elif self.loss_function is LossFunction.Huber:
            delta = 1.0  
            mask = np.abs(e) <= delta
            grad_component = np.where(mask, e, delta * np.sign(e))
            gradient = (1 / l) * (X.T @ grad_component) + 2 * self.l2_coef * self.w

        else:
            raise NotImplementedError(":((")
        
        return gradient


    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        l = X.shape[0]
        e = X @ self.w - y

        if self.loss_function is LossFunction.MSE:
            loss = (1 / l) * np.sum(e ** 2) + self.l2_coef * np.sum(self.w ** 2)

        elif self.loss_function is LossFunction.MAE:
            loss = (1 / l) * np.sum(np.abs(e)) + self.l2_coef * np.sum(self.w ** 2)

        elif self.loss_function is LossFunction.LogCosh:
            loss = (1 / l) * np.sum(np.log(np.cosh(e))) + self.l2_coef * np.sum(self.w ** 2)

        elif self.loss_function is LossFunction.Huber:
            delta = 1.0
            mask = np.abs(e) <= delta
            loss = (1 / l) * np.sum(
                np.where(mask, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))) + self.l2_coef * np.sum(self.w ** 2)

        else:
            raise NotImplementedError(":((")
        return loss


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.y_train = X, y

        if self.optimizer is None:
            I = np.eye(X.shape[1])
            self.w = np.linalg.inv(X.T @ X + self.l2_coef * I) @ X.T @ y
            self.loss_history = [self.compute_loss(X, y)]
            return self

        elif self.optimizer == "SVD":
            # усечённое SVD с 4 компонентами
            U, s, Vt = svds(X, k=4)
            Σ_inv = np.diag(1 / s)
            self.w = Vt.T @ Σ_inv @ U.T @ y
            self.loss_history = [self.compute_loss(X, y)]
            return self

        elif isinstance(self.optimizer, BaseDescent):
            self.w = np.zeros(X.shape[1])   
            for _ in range(self.max_iter):
                delta_w = self.optimizer.step()
                self.loss_history.append(self.compute_loss(X, y))
                if np.any(np.isnan(delta_w)):
                    break
                if np.linalg.norm(delta_w) ** 2 < self.tolerance:
                    break
            return self

        else:
            raise ValueError("Unknown optimizer type")


