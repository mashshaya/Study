from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_multiple_brownian_motion_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_value: float = 0.0,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate standard Brownian motion paths on an equally spaced time grid."""

    if number_of_paths <= 0 or number_of_time_steps <= 0:
        raise ValueError("number_of_paths and number_of_time_steps must be positive.")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive.")

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    increments = rng.normal(
        loc=0.0,
        scale=np.sqrt(time_step),
        size=(number_of_paths, number_of_time_steps),
    )

    simulated_paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    simulated_paths[:, 0] = initial_value
    simulated_paths[:, 1:] = initial_value + np.cumsum(increments, axis=1)
    return time_grid, simulated_paths


def simulate_multiple_brownian_motion_paths_with_drift_and_volatility(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    drift: float,
    volatility: float,
    initial_value: float = 0.0,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an arithmetic Brownian motion X_t = X_0 + mu t + sigma W_t."""

    if volatility < 0.0:
        raise ValueError("volatility must be non-negative.")

    time_grid, brownian_paths = simulate_multiple_brownian_motion_paths(
        number_of_paths=number_of_paths,
        number_of_time_steps=number_of_time_steps,
        final_time=final_time,
        initial_value=0.0,
        random_seed=random_seed,
    )
    simulated_paths = initial_value + drift * time_grid[None, :] + volatility * brownian_paths
    return time_grid, simulated_paths


def calculate_empirical_mean_and_variance_across_paths(
    simulated_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical mean and variance across paths at every time point."""

    empirical_mean = simulated_paths.mean(axis=0)
    empirical_variance = simulated_paths.var(axis=0)
    return empirical_mean, empirical_variance


def calculate_theoretical_mean_and_variance_for_brownian_motion(
    time_grid: np.ndarray,
    initial_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E[W_t] and Var(W_t) for standard Brownian motion."""

    theoretical_mean = np.full_like(time_grid, initial_value, dtype=float)
    theoretical_variance = time_grid.copy()
    return theoretical_mean, theoretical_variance


def calculate_theoretical_mean_and_variance_for_arithmetic_brownian_motion(
    time_grid: np.ndarray,
    drift: float,
    volatility: float,
    initial_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and variance of X_t = X_0 + mu t + sigma W_t."""

    theoretical_mean = initial_value + drift * time_grid
    theoretical_variance = volatility**2 * time_grid
    return theoretical_mean, theoretical_variance


def extract_brownian_motion_increments_over_interval(
    simulated_paths: np.ndarray,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Extract pathwise Brownian increments W_t - W_s over a grid interval."""

    return simulated_paths[:, end_index] - simulated_paths[:, start_index]


def calculate_empirical_correlation_between_two_brownian_increments(
    simulated_paths: np.ndarray,
    first_interval_start: int,
    first_interval_end: int,
    second_interval_start: int,
    second_interval_end: int,
) -> float:
    """Estimate correlation between two Brownian increments."""

    first_increment = extract_brownian_motion_increments_over_interval(
        simulated_paths,
        first_interval_start,
        first_interval_end,
    )
    second_increment = extract_brownian_motion_increments_over_interval(
        simulated_paths,
        second_interval_start,
        second_interval_end,
    )
    correlation_matrix = np.corrcoef(first_increment, second_increment)
    return float(correlation_matrix[0, 1])


def calculate_empirical_covariance_matrix_at_selected_times(
    time_grid: np.ndarray,
    simulated_paths: np.ndarray,
    selected_times: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical covariance matrix at selected time points."""

    selected_indices = np.array(
        [int(np.argmin(np.abs(time_grid - selected_time))) for selected_time in selected_times]
    )
    selected_path_values = simulated_paths[:, selected_indices]
    empirical_covariance_matrix = np.cov(selected_path_values, rowvar=False, bias=True)
    return selected_indices, empirical_covariance_matrix


def calculate_theoretical_brownian_covariance_matrix(
    selected_times: np.ndarray,
) -> np.ndarray:
    """Return the Brownian covariance matrix Cov(W_s, W_t) = min(s, t)."""

    return np.minimum.outer(selected_times, selected_times)


def calculate_quadratic_variation_for_each_path(
    simulated_paths: np.ndarray,
) -> np.ndarray:
    """Calculate sum of squared increments for every simulated path."""

    increments = np.diff(simulated_paths, axis=1)
    return np.sum(increments**2, axis=1)


def calculate_total_variation_for_each_path(
    simulated_paths: np.ndarray,
) -> np.ndarray:
    """Calculate discrete total variation for every simulated path."""

    increments = np.diff(simulated_paths, axis=1)
    return np.sum(np.abs(increments), axis=1)


def plot_mean_quadratic_and_total_variation_by_grid_size(
    number_of_time_steps: np.ndarray,
    mean_quadratic_variations: np.ndarray,
    mean_total_variations: np.ndarray,
    final_time: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Compare quadratic and total variation estimates as grid is refined."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        number_of_time_steps,
        mean_quadratic_variations,
        marker="o",
        linewidth=2.0,
        label="Средняя квадратичная вариация",
    )
    axis.plot(
        number_of_time_steps,
        mean_total_variations,
        marker="o",
        linewidth=2.0,
        label="Средняя полная вариация",
    )
    axis.axhline(final_time, color="black", linestyle="--", linewidth=1.2, label=f"Теория [W] = {final_time}")
    axis.set_xscale("log")
    axis.set_title(title)
    axis.set_xlabel("Число шагов сетки")
    axis.set_ylabel("Среднее значение по траекториям")
    axis.legend()
    axis.grid(alpha=0.3, which="both")
    return figure, axis


def estimate_first_passage_times_for_brownian_motion_paths(
    time_grid: np.ndarray,
    simulated_paths: np.ndarray,
    upper_barrier: float,
) -> np.ndarray:
    """Estimate first time when each path reaches or crosses an upper barrier."""

    if upper_barrier <= 0.0:
        raise ValueError("upper_barrier must be positive.")

    crossed_barrier = simulated_paths >= upper_barrier
    first_passage_times = np.full(simulated_paths.shape[0], np.nan, dtype=float)
    for path_index, path_crossings in enumerate(crossed_barrier):
        if path_crossings.any():
            first_passage_times[path_index] = time_grid[int(np.argmax(path_crossings))]
    return first_passage_times


def calculate_theoretical_brownian_first_passage_probability_before_time(
    upper_barrier: float,
    final_time: float,
) -> float:
    """Return P(tau_a <= T) for standard Brownian motion using reflection principle."""

    if upper_barrier <= 0.0 or final_time <= 0.0:
        raise ValueError("upper_barrier and final_time must be positive.")
    return float(2.0 * (1.0 - norm.cdf(upper_barrier / np.sqrt(final_time))))


def plot_brownian_motion_sample_paths(
    time_grid: np.ndarray,
    simulated_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot a subset of Brownian motion paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for path_number in range(min(number_of_paths_to_plot, simulated_paths.shape[0])):
        axis.plot(
            time_grid,
            simulated_paths[path_number],
            linewidth=1.2,
            alpha=0.85,
            label=f"Траектория {path_number + 1}" if path_number < 6 else None,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Значение процесса")
    if min(number_of_paths_to_plot, simulated_paths.shape[0]) <= 6:
        axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_empirical_and_theoretical_brownian_mean_and_variance(
    time_grid: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_variance: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_variance: np.ndarray,
    title_prefix: str,
) -> tuple[Figure, np.ndarray]:
    """Plot empirical and theoretical Brownian mean and variance."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_grid, empirical_mean, label="Эмпирическое среднее", linewidth=2.0)
    axes[0].plot(
        time_grid,
        theoretical_mean,
        label="Теоретическое среднее",
        linewidth=2.0,
        linestyle="--",
    )
    axes[0].set_title(f"{title_prefix}: среднее")
    axes[0].set_xlabel("Время")
    axes[0].set_ylabel("Среднее значение")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_grid, empirical_variance, label="Эмпирическая дисперсия", linewidth=2.0)
    axes[1].plot(
        time_grid,
        theoretical_variance,
        label="Теоретическая дисперсия",
        linewidth=2.0,
        linestyle="--",
    )
    axes[1].set_title(f"{title_prefix}: дисперсия")
    axes[1].set_xlabel("Время")
    axes[1].set_ylabel("Дисперсия")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


def plot_brownian_increment_distribution_with_normal_density(
    increments: np.ndarray,
    interval_length: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical Brownian increments with their theoretical normal density."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(increments, bins=45, density=True, alpha=0.55, color="#4c78a8", label="Симуляция")
    dense_grid = np.linspace(increments.min() - 0.2, increments.max() + 0.2, 500)
    axis.plot(
        dense_grid,
        norm.pdf(dense_grid, loc=0.0, scale=np.sqrt(interval_length)),
        color="#f58518",
        linewidth=2.3,
        label=rf"$N(0, {interval_length:.2f})$",
    )
    axis.set_title(title)
    axis.set_xlabel("Приращение")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_covariance_matrix_heatmaps(
    selected_times: np.ndarray,
    empirical_covariance_matrix: np.ndarray,
    theoretical_covariance_matrix: np.ndarray,
) -> tuple[Figure, np.ndarray]:
    """Plot empirical and theoretical Brownian covariance matrices."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    matrices = [
        (empirical_covariance_matrix, "Эмпирическая ковариация"),
        (theoretical_covariance_matrix, "Теоретическая ковариация min(s, t)"),
    ]
    color_limit = max(
        float(np.abs(empirical_covariance_matrix).max()),
        float(np.abs(theoretical_covariance_matrix).max()),
    )
    for axis, (matrix, title) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix, vmin=0.0, vmax=color_limit, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(len(selected_times)))
        axis.set_yticks(range(len(selected_times)))
        axis.set_xticklabels([f"{time:.2f}" for time in selected_times])
        axis.set_yticklabels([f"{time:.2f}" for time in selected_times])
        axis.set_xlabel("t")
        axis.set_ylabel("s")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure, axes


def plot_quadratic_variation_distribution(
    quadratic_variations: np.ndarray,
    final_time: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot distribution of quadratic variation estimates."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(quadratic_variations, bins=40, color="#54a24b", alpha=0.75, edgecolor="white")
    axis.axvline(final_time, color="black", linestyle="--", linewidth=2.0, label=f"Теория: {final_time}")
    axis.set_title(title)
    axis.set_xlabel("Квадратичная вариация")
    axis.set_ylabel("Число траекторий")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_first_passage_time_histogram(
    first_passage_times: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot first passage times among paths that crossed the barrier."""

    valid_first_passage_times = first_passage_times[~np.isnan(first_passage_times)]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(
        valid_first_passage_times,
        bins=35,
        color="#e45756",
        alpha=0.75,
        edgecolor="white",
    )
    axis.set_title(title)
    axis.set_xlabel("Оценённое время первого достижения")
    axis.set_ylabel("Число траекторий")
    axis.grid(alpha=0.3)
    return figure, axis


def summarize_first_passage_simulation(
    first_passage_times: np.ndarray,
    final_time: float,
    upper_barrier: float,
) -> dict[str, Any]:
    """Summarize empirical and theoretical first passage results."""

    valid_first_passage_times = first_passage_times[~np.isnan(first_passage_times)]
    return {
        "fraction_crossed_before_final_time": float(np.mean(~np.isnan(first_passage_times))),
        "theoretical_fraction_crossed_before_final_time": calculate_theoretical_brownian_first_passage_probability_before_time(
            upper_barrier=upper_barrier,
            final_time=final_time,
        ),
        "mean_first_passage_time_among_crossed_paths": float(np.mean(valid_first_passage_times)),
        "median_first_passage_time_among_crossed_paths": float(np.median(valid_first_passage_times)),
        "number_of_crossed_paths": int(valid_first_passage_times.size),
    }
