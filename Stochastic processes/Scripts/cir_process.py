from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gamma


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_cir_paths_full_truncation_euler(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate CIR paths using full truncation Euler."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for step in range(number_of_time_steps):
        positive_state = np.maximum(paths[:, step], 0.0)
        drift = mean_reversion_speed * (long_term_mean - positive_state) * time_step
        diffusion = volatility * np.sqrt(positive_state * time_step) * rng.normal(size=number_of_paths)
        paths[:, step + 1] = np.maximum(paths[:, step] + drift + diffusion, 0.0)
    return time_grid, paths


def simulate_cir_paths_plain_euler(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate CIR paths with plain Euler, allowing negative values."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for step in range(number_of_time_steps):
        sqrt_state = np.sqrt(np.maximum(paths[:, step], 0.0))
        drift = mean_reversion_speed * (long_term_mean - paths[:, step]) * time_step
        diffusion = volatility * sqrt_state * np.sqrt(time_step) * rng.normal(size=number_of_paths)
        paths[:, step + 1] = paths[:, step] + drift + diffusion
    return time_grid, paths


def calculate_theoretical_cir_mean_and_variance(
    time_grid: np.ndarray,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical CIR mean and variance."""

    decay = np.exp(-mean_reversion_speed * time_grid)
    mean = long_term_mean + (initial_value - long_term_mean) * decay
    variance = (
        initial_value * volatility**2 * decay * (1.0 - decay) / mean_reversion_speed
        + long_term_mean * volatility**2 * (1.0 - decay) ** 2 / (2.0 * mean_reversion_speed)
    )
    return mean, variance


def calculate_cir_feller_ratio(
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
) -> float:
    """Return Feller ratio 2 kappa theta / sigma^2."""

    return float(2.0 * mean_reversion_speed * long_term_mean / volatility**2)


def calculate_cir_stationary_gamma_parameters(
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
) -> tuple[float, float]:
    """Return shape and scale of stationary CIR gamma distribution."""

    shape = 2.0 * mean_reversion_speed * long_term_mean / volatility**2
    scale = volatility**2 / (2.0 * mean_reversion_speed)
    return float(shape), float(scale)


def plot_cir_paths(
    time_grid: np.ndarray,
    paths: np.ndarray,
    theoretical_mean: np.ndarray,
    long_term_mean: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot CIR paths with mean levels."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(12, paths.shape[0])):
        axis.plot(time_grid, paths[index], linewidth=1.2, alpha=0.85)
    axis.plot(time_grid, theoretical_mean, color="black", linewidth=2.4, label="E[X(t)]")
    axis.axhline(long_term_mean, color="#e45756", linestyle="--", label="theta")
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("X(t)")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_cir_mean_variance_comparison(
    time_grid: np.ndarray,
    paths: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_variance: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot CIR empirical and theoretical mean and variance."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_grid, paths.mean(axis=0), label="Симуляция", linewidth=2.0)
    axes[0].plot(time_grid, theoretical_mean, "--", label="Теория", linewidth=2.0)
    axes[0].set_title(f"{title}: среднее")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_grid, paths.var(axis=0), label="Симуляция", linewidth=2.0)
    axes[1].plot(time_grid, theoretical_variance, "--", label="Теория", linewidth=2.0)
    axes[1].set_title(f"{title}: дисперсия")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


def plot_cir_stationary_distribution(
    terminal_values: np.ndarray,
    shape: float,
    scale: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot CIR terminal values against stationary gamma density."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(terminal_values, bins=55, density=True, alpha=0.55, color="#4c78a8", label="Симуляция")
    grid = np.linspace(0.0, np.quantile(terminal_values, 0.999), 600)
    axis.plot(grid, gamma.pdf(grid, a=shape, scale=scale), color="#f58518", linewidth=2.2, label="Стационарная gamma")
    axis.set_title(title)
    axis.set_xlabel("X(T)")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
