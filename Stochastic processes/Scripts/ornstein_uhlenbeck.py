from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_multiple_ornstein_uhlenbeck_paths_exact(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate OU paths exactly on an equally spaced grid."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    decay = np.exp(-mean_reversion_speed * time_step)
    innovation_std = volatility * np.sqrt((1.0 - decay**2) / (2.0 * mean_reversion_speed))
    paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for step in range(number_of_time_steps):
        conditional_mean = long_term_mean + (paths[:, step] - long_term_mean) * decay
        paths[:, step + 1] = conditional_mean + innovation_std * rng.normal(size=number_of_paths)
    return time_grid, paths


def simulate_multiple_ornstein_uhlenbeck_paths_euler(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate OU paths with the Euler-Maruyama scheme."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for step in range(number_of_time_steps):
        drift = mean_reversion_speed * (long_term_mean - paths[:, step]) * time_step
        diffusion = volatility * np.sqrt(time_step) * rng.normal(size=number_of_paths)
        paths[:, step + 1] = paths[:, step] + drift + diffusion
    return time_grid, paths


def calculate_theoretical_ornstein_uhlenbeck_mean_and_variance(
    time_grid: np.ndarray,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical mean and variance of an OU process."""

    decay = np.exp(-mean_reversion_speed * time_grid)
    mean = long_term_mean + (initial_value - long_term_mean) * decay
    variance = volatility**2 / (2.0 * mean_reversion_speed) * (1.0 - decay**2)
    return mean, variance


def calculate_ornstein_uhlenbeck_stationary_standard_deviation(
    mean_reversion_speed: float,
    volatility: float,
) -> float:
    """Return stationary standard deviation sigma / sqrt(2 kappa)."""

    return float(volatility / np.sqrt(2.0 * mean_reversion_speed))


def calculate_ornstein_uhlenbeck_half_life(mean_reversion_speed: float) -> float:
    """Return half-life of a deterministic OU deviation."""

    return float(np.log(2.0) / mean_reversion_speed)


def calculate_empirical_mean_variance_and_autocorrelation(
    paths: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate pathwise mean, variance, and lag autocorrelation."""

    mean = paths.mean(axis=0)
    variance = paths.var(axis=0)
    flattened_left = paths[:, :-lag].ravel()
    flattened_right = paths[:, lag:].ravel()
    autocorrelation = float(np.corrcoef(flattened_left, flattened_right)[0, 1])
    return mean, variance, autocorrelation


def plot_ornstein_uhlenbeck_paths_with_mean(
    time_grid: np.ndarray,
    paths: np.ndarray,
    theoretical_mean: np.ndarray,
    long_term_mean: float,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot OU sample paths with theoretical and long-term mean."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(number_of_paths_to_plot, paths.shape[0])):
        axis.plot(time_grid, paths[index], alpha=0.8, linewidth=1.2)
    axis.plot(time_grid, theoretical_mean, color="black", linewidth=2.5, label="E[X(t)]")
    axis.axhline(long_term_mean, color="#e45756", linestyle="--", linewidth=2.0, label="Долгосрочное среднее")
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("X(t)")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_ornstein_uhlenbeck_mean_variance_comparison(
    time_grid: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_variance: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_variance: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot empirical and theoretical OU mean and variance."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_grid, empirical_mean, label="Симуляция", linewidth=2.0)
    axes[0].plot(time_grid, theoretical_mean, "--", label="Теория", linewidth=2.0)
    axes[0].set_title(f"{title}: среднее")
    axes[0].set_xlabel("Время")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_grid, empirical_variance, label="Симуляция", linewidth=2.0)
    axes[1].plot(time_grid, theoretical_variance, "--", label="Теория", linewidth=2.0)
    axes[1].set_title(f"{title}: дисперсия")
    axes[1].set_xlabel("Время")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


def plot_ornstein_uhlenbeck_stationary_distribution(
    terminal_values: np.ndarray,
    long_term_mean: float,
    stationary_std: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical terminal OU distribution with stationary normal density."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(terminal_values, bins=55, density=True, alpha=0.55, color="#4c78a8", label="Симуляция")
    grid = np.linspace(np.quantile(terminal_values, 0.001), np.quantile(terminal_values, 0.999), 600)
    axis.plot(grid, norm.pdf(grid, long_term_mean, stationary_std), color="#f58518", linewidth=2.2, label="Стационарная плотность")
    axis.set_title(title)
    axis.set_xlabel("X(T)")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_ornstein_uhlenbeck_parameter_scenarios(
    scenario_paths: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot mean paths for several OU parameter scenarios."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for scenario_name, (time_grid, paths) in scenario_paths.items():
        axis.plot(time_grid, paths.mean(axis=0), linewidth=2.0, label=scenario_name)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Среднее по траекториям")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
