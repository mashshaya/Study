from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import lognorm, norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_multiple_geometric_brownian_motion_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_price: float,
    drift: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GBM exactly on an equally spaced time grid."""

    if number_of_paths <= 0 or number_of_time_steps <= 0:
        raise ValueError("number_of_paths and number_of_time_steps must be positive.")
    if final_time <= 0.0 or initial_price <= 0.0 or volatility < 0.0:
        raise ValueError("final_time and initial_price must be positive; volatility non-negative.")

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    normal_increments = rng.normal(size=(number_of_paths, number_of_time_steps))
    log_increments = (drift - 0.5 * volatility**2) * time_step + volatility * np.sqrt(time_step) * normal_increments
    log_paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(initial_price)
    log_paths[:, 1:] = np.log(initial_price) + np.cumsum(log_increments, axis=1)
    return time_grid, np.exp(log_paths)


def calculate_theoretical_gbm_mean_variance_and_median(
    time_grid: np.ndarray,
    initial_price: float,
    drift: float,
    volatility: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return theoretical mean, variance, and median of GBM."""

    mean = initial_price * np.exp(drift * time_grid)
    variance = (
        initial_price**2
        * np.exp(2.0 * drift * time_grid)
        * (np.exp(volatility**2 * time_grid) - 1.0)
    )
    median = initial_price * np.exp((drift - 0.5 * volatility**2) * time_grid)
    return mean, variance, median


def calculate_empirical_mean_variance_and_median_across_paths(
    simulated_price_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate empirical mean, variance, and median of price paths."""

    return (
        simulated_price_paths.mean(axis=0),
        simulated_price_paths.var(axis=0),
        np.median(simulated_price_paths, axis=0),
    )


def calculate_log_returns_from_price_paths(
    simulated_price_paths: np.ndarray,
) -> np.ndarray:
    """Calculate one-step log returns from simulated price paths."""

    return np.diff(np.log(simulated_price_paths), axis=1)


def calculate_theoretical_probability_of_terminal_price_exceeding_level(
    initial_price: float,
    drift: float,
    volatility: float,
    final_time: float,
    price_level: float,
) -> float:
    """Return P(S_T >= level) under GBM."""

    if volatility == 0.0:
        deterministic_price = initial_price * np.exp(drift * final_time)
        return float(deterministic_price >= price_level)
    z_value = (
        np.log(price_level / initial_price)
        - (drift - 0.5 * volatility**2) * final_time
    ) / (volatility * np.sqrt(final_time))
    return float(1.0 - norm.cdf(z_value))


def estimate_probability_of_reaching_price_barrier_before_maturity(
    simulated_price_paths: np.ndarray,
    price_barrier: float,
) -> float:
    """Estimate probability of ever reaching a price barrier on the simulated grid."""

    return float(np.mean(np.max(simulated_price_paths, axis=1) >= price_barrier))


def plot_geometric_brownian_motion_sample_paths(
    time_grid: np.ndarray,
    simulated_price_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot sample GBM price paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for path_index in range(min(number_of_paths_to_plot, simulated_price_paths.shape[0])):
        axis.plot(time_grid, simulated_price_paths[path_index], linewidth=1.2, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Цена S(t)")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_gbm_mean_median_and_simulation(
    time_grid: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_median: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_median: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical and theoretical GBM mean and median."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(time_grid, empirical_mean, label="Эмпирическое среднее", linewidth=2.0)
    axis.plot(time_grid, theoretical_mean, "--", label="Теоретическое среднее", linewidth=2.0)
    axis.plot(time_grid, empirical_median, label="Эмпирическая медиана", linewidth=2.0)
    axis.plot(time_grid, theoretical_median, "--", label="Теоретическая медиана", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Цена")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_terminal_gbm_distribution_with_lognormal_density(
    terminal_prices: np.ndarray,
    initial_price: float,
    drift: float,
    volatility: float,
    final_time: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot terminal GBM prices against their lognormal density."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(terminal_prices, bins=60, density=True, alpha=0.55, color="#4c78a8", label="Симуляция")
    dense_grid = np.linspace(np.quantile(terminal_prices, 0.001), np.quantile(terminal_prices, 0.999), 700)
    scale = initial_price * np.exp((drift - 0.5 * volatility**2) * final_time)
    axis.plot(
        dense_grid,
        lognorm.pdf(dense_grid, s=volatility * np.sqrt(final_time), scale=scale),
        color="#f58518",
        linewidth=2.2,
        label="Логнормальная плотность",
    )
    axis.set_title(title)
    axis.set_xlabel("S(T)")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_gbm_parameter_scenarios(
    scenario_results: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot theoretical mean paths for several GBM parameter scenarios."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for scenario_name, (time_grid, mean_path) in scenario_results.items():
        axis.plot(time_grid, mean_path, linewidth=2.0, label=scenario_name)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Теоретическое E[S(t)]")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_gbm_log_return_distribution_with_normal_density(
    log_returns: np.ndarray,
    drift: float,
    volatility: float,
    time_step: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot one-step GBM log returns against their normal density."""

    flattened_log_returns = log_returns.ravel()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(flattened_log_returns, bins=60, density=True, alpha=0.55, color="#54a24b", label="Симуляция")
    dense_grid = np.linspace(np.quantile(flattened_log_returns, 0.001), np.quantile(flattened_log_returns, 0.999), 700)
    axis.plot(
        dense_grid,
        norm.pdf(
            dense_grid,
            loc=(drift - 0.5 * volatility**2) * time_step,
            scale=volatility * np.sqrt(time_step),
        ),
        color="#e45756",
        linewidth=2.2,
        label="Нормальная плотность",
    )
    axis.set_title(title)
    axis.set_xlabel("Лог-доходность за шаг")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def calculate_discounted_price_paths(
    simulated_price_paths: np.ndarray,
    time_grid: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    """Discount price paths by exp(-r t)."""

    return simulated_price_paths * np.exp(-risk_free_rate * time_grid[None, :])
