from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import kurtosis


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_merton_jump_diffusion_price_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_price: float,
    drift: float,
    volatility: float,
    jump_intensity: float,
    jump_log_mean: float,
    jump_log_std: float,
    compensate_jump_drift: bool = True,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Merton jump-diffusion price paths."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    dt = final_time / number_of_time_steps
    jump_compensator = jump_intensity * (np.exp(jump_log_mean + 0.5 * jump_log_std**2) - 1.0)
    effective_drift = drift - jump_compensator if compensate_jump_drift else drift
    normal_shocks = rng.normal(size=(number_of_paths, number_of_time_steps))
    jump_counts = rng.poisson(jump_intensity * dt, size=(number_of_paths, number_of_time_steps))
    jump_log_sums = np.zeros_like(normal_shocks)
    for path_index, step_index in zip(*np.nonzero(jump_counts)):
        jump_log_sums[path_index, step_index] = rng.normal(
            jump_log_mean,
            jump_log_std,
            size=jump_counts[path_index, step_index],
        ).sum()
    log_increments = (
        (effective_drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * normal_shocks
        + jump_log_sums
    )
    log_paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(initial_price)
    log_paths[:, 1:] = np.log(initial_price) + np.cumsum(log_increments, axis=1)
    return time_grid, np.exp(log_paths), jump_counts


def calculate_log_returns_from_price_paths(price_paths: np.ndarray) -> np.ndarray:
    """Calculate one-step log returns."""

    return np.diff(np.log(price_paths), axis=1)


def calculate_maximum_drawdowns(price_paths: np.ndarray) -> np.ndarray:
    """Calculate maximum drawdown for each path."""

    running_maximum = np.maximum.accumulate(price_paths, axis=1)
    drawdowns = price_paths / running_maximum - 1.0
    return drawdowns.min(axis=1)


def summarize_jump_diffusion_risk_statistics(
    price_paths: np.ndarray,
    jump_counts: np.ndarray,
    loss_threshold: float,
) -> dict[str, float]:
    """Summarize terminal and path-dependent jump-diffusion risk statistics."""

    log_returns = calculate_log_returns_from_price_paths(price_paths).ravel()
    terminal_returns = price_paths[:, -1] / price_paths[:, 0] - 1.0
    maximum_drawdowns = calculate_maximum_drawdowns(price_paths)
    return {
        "mean_terminal_return": float(terminal_returns.mean()),
        "terminal_return_std": float(terminal_returns.std()),
        "probability_terminal_loss_below_threshold": float(np.mean(terminal_returns <= loss_threshold)),
        "log_return_excess_kurtosis": float(kurtosis(log_returns, fisher=True, bias=False)),
        "mean_total_jump_count": float(jump_counts.sum(axis=1).mean()),
        "mean_maximum_drawdown": float(maximum_drawdowns.mean()),
        "five_percent_worst_drawdown": float(np.quantile(maximum_drawdowns, 0.05)),
    }


def plot_price_paths(
    time_grid: np.ndarray,
    price_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot sample price paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(number_of_paths_to_plot, price_paths.shape[0])):
        axis.plot(time_grid, price_paths[index], linewidth=1.1, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Цена")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_log_return_histograms(
    gbm_log_returns: np.ndarray,
    jump_log_returns: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Compare GBM and jump-diffusion log-return histograms."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(gbm_log_returns.ravel(), bins=80, density=True, alpha=0.5, label="GBM")
    axis.hist(jump_log_returns.ravel(), bins=80, density=True, alpha=0.5, label="Jump-diffusion")
    axis.set_title(title)
    axis.set_xlabel("Лог-доходность за шаг")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_drawdown_distribution(
    drawdowns_by_name: dict[str, np.ndarray],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot maximum drawdown distributions."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, drawdowns in drawdowns_by_name.items():
        axis.hist(drawdowns, bins=50, alpha=0.55, density=True, label=name)
    axis.set_title(title)
    axis.set_xlabel("Максимальная просадка")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
