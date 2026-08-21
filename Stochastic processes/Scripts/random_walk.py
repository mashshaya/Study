from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import binom, norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_multiple_random_walk_paths(
    number_of_paths: int,
    number_of_steps: int,
    probability_of_upward_step: float,
    upward_step_size: float = 1.0,
    downward_step_size: float | None = None,
    initial_position: float = 0.0,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate multiple discrete-time random walk paths."""

    if not 0.0 <= probability_of_upward_step <= 1.0:
        raise ValueError("probability_of_upward_step must lie in [0, 1].")
    if number_of_paths <= 0 or number_of_steps <= 0:
        raise ValueError("number_of_paths and number_of_steps must be positive.")

    downward_step_size = upward_step_size if downward_step_size is None else downward_step_size
    rng = create_random_number_generator(random_seed)
    upward_indicators = (
        rng.random(size=(number_of_paths, number_of_steps)) < probability_of_upward_step
    )
    increments = np.where(upward_indicators, upward_step_size, -downward_step_size)

    simulated_paths = np.empty((number_of_paths, number_of_steps + 1), dtype=float)
    simulated_paths[:, 0] = initial_position
    simulated_paths[:, 1:] = initial_position + np.cumsum(increments, axis=1)
    step_index = np.arange(number_of_steps + 1)
    return step_index, simulated_paths


def calculate_theoretical_mean_and_variance_of_random_walk(
    step_index: np.ndarray,
    probability_of_upward_step: float,
    upward_step_size: float = 1.0,
    downward_step_size: float | None = None,
    initial_position: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical mean and variance of a random walk at each step."""

    downward_step_size = upward_step_size if downward_step_size is None else downward_step_size
    mean_of_one_increment = (
        probability_of_upward_step * upward_step_size
        - (1.0 - probability_of_upward_step) * downward_step_size
    )
    second_moment_of_one_increment = (
        probability_of_upward_step * upward_step_size**2
        + (1.0 - probability_of_upward_step) * downward_step_size**2
    )
    variance_of_one_increment = second_moment_of_one_increment - mean_of_one_increment**2

    theoretical_mean = initial_position + step_index * mean_of_one_increment
    theoretical_variance = step_index * variance_of_one_increment
    return theoretical_mean, theoretical_variance


def calculate_empirical_mean_and_variance_across_paths(
    simulated_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical mean and variance across paths at every step."""

    empirical_mean = simulated_paths.mean(axis=0)
    empirical_variance = simulated_paths.var(axis=0)
    return empirical_mean, empirical_variance


def calculate_empirical_correlation_between_two_non_overlapping_random_walk_increments(
    simulated_paths: np.ndarray,
    first_interval_start: int,
    first_interval_end: int,
    second_interval_start: int,
    second_interval_end: int,
) -> float:
    """Estimate the correlation between two non-overlapping increments."""

    first_increment = (
        simulated_paths[:, first_interval_end] - simulated_paths[:, first_interval_start]
    )
    second_increment = (
        simulated_paths[:, second_interval_end] - simulated_paths[:, second_interval_start]
    )
    correlation_matrix = np.corrcoef(first_increment, second_increment)
    return float(correlation_matrix[0, 1])


def calculate_terminal_position_values_and_probabilities(
    number_of_steps: int,
    probability_of_upward_step: float,
    upward_step_size: float = 1.0,
    downward_step_size: float | None = None,
    initial_position: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact terminal positions and their probabilities."""

    downward_step_size = upward_step_size if downward_step_size is None else downward_step_size
    number_of_upward_moves = np.arange(number_of_steps + 1)
    terminal_positions = (
        initial_position
        + number_of_upward_moves * upward_step_size
        - (number_of_steps - number_of_upward_moves) * downward_step_size
    )
    terminal_probabilities = binom.pmf(
        number_of_upward_moves,
        n=number_of_steps,
        p=probability_of_upward_step,
    )
    return terminal_positions, terminal_probabilities


def calculate_normal_approximation_density_for_terminal_position(
    x_values: np.ndarray,
    number_of_steps: int,
    probability_of_upward_step: float,
    upward_step_size: float = 1.0,
    downward_step_size: float | None = None,
    initial_position: float = 0.0,
) -> np.ndarray:
    """Evaluate the CLT-based normal approximation for terminal position."""

    theoretical_mean, theoretical_variance = calculate_theoretical_mean_and_variance_of_random_walk(
        step_index=np.array([number_of_steps]),
        probability_of_upward_step=probability_of_upward_step,
        upward_step_size=upward_step_size,
        downward_step_size=downward_step_size,
        initial_position=initial_position,
    )
    standard_deviation = np.sqrt(theoretical_variance[0])
    if standard_deviation == 0.0:
        return np.zeros_like(x_values, dtype=float)
    return norm.pdf(x_values, loc=theoretical_mean[0], scale=standard_deviation)


def plot_random_walk_sample_paths(
    step_index: np.ndarray,
    simulated_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot a subset of random walk paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for path_number in range(min(number_of_paths_to_plot, simulated_paths.shape[0])):
        axis.plot(
            step_index,
            simulated_paths[path_number],
            linewidth=1.2,
            alpha=0.85,
            label=f"Траектория {path_number + 1}" if path_number < 6 else None,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel("Номер шага")
    axis.set_ylabel("Положение процесса")
    if min(number_of_paths_to_plot, simulated_paths.shape[0]) <= 6:
        axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_empirical_and_theoretical_mean_and_variance(
    step_index: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_variance: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_variance: np.ndarray,
    title_prefix: str,
) -> tuple[Figure, np.ndarray]:
    """Compare empirical and theoretical mean and variance."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(step_index, empirical_mean, label="Эмпирическое среднее", linewidth=2.0)
    axes[0].plot(
        step_index,
        theoretical_mean,
        label="Теоретическое среднее",
        linewidth=2.0,
        linestyle="--",
    )
    axes[0].set_title(f"{title_prefix}: среднее")
    axes[0].set_xlabel("Номер шага")
    axes[0].set_ylabel("Среднее положение")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(step_index, empirical_variance, label="Эмпирическая дисперсия", linewidth=2.0)
    axes[1].plot(
        step_index,
        theoretical_variance,
        label="Теоретическая дисперсия",
        linewidth=2.0,
        linestyle="--",
    )
    axes[1].set_title(f"{title_prefix}: дисперсия")
    axes[1].set_xlabel("Номер шага")
    axes[1].set_ylabel("Дисперсия положения")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    return figure, axes


def plot_terminal_distribution_with_theory_and_normal_approximation(
    terminal_positions_from_simulation: np.ndarray,
    number_of_steps: int,
    probability_of_upward_step: float,
    upward_step_size: float = 1.0,
    downward_step_size: float | None = None,
    initial_position: float = 0.0,
    title: str = "Распределение конечного положения случайного блуждания",
) -> tuple[Figure, Axes]:
    """Plot empirical terminal distribution, exact theory, and normal approximation."""

    figure, axis = plt.subplots(figsize=(11, 6))
    unique_positions = np.sort(np.unique(terminal_positions_from_simulation))
    histogram_weights = np.full(terminal_positions_from_simulation.shape[0], 1.0 / terminal_positions_from_simulation.shape[0])
    bins = np.arange(unique_positions.min() - 1, unique_positions.max() + 2)
    axis.hist(
        terminal_positions_from_simulation,
        bins=bins,
        weights=histogram_weights,
        alpha=0.45,
        color="#4c78a8",
        label="Эмпирическое распределение",
    )

    terminal_positions, terminal_probabilities = calculate_terminal_position_values_and_probabilities(
        number_of_steps=number_of_steps,
        probability_of_upward_step=probability_of_upward_step,
        upward_step_size=upward_step_size,
        downward_step_size=downward_step_size,
        initial_position=initial_position,
    )
    axis.stem(
        terminal_positions,
        terminal_probabilities,
        linefmt="#f58518",
        markerfmt="o",
        basefmt=" ",
        label="Точная теория",
    )

    dense_grid = np.linspace(terminal_positions.min() - 3, terminal_positions.max() + 3, 500)
    normal_density = calculate_normal_approximation_density_for_terminal_position(
        x_values=dense_grid,
        number_of_steps=number_of_steps,
        probability_of_upward_step=probability_of_upward_step,
        upward_step_size=upward_step_size,
        downward_step_size=downward_step_size,
        initial_position=initial_position,
    )
    axis.plot(
        dense_grid,
        normal_density,
        color="#54a24b",
        linewidth=2.2,
        label="Нормальная аппроксимация",
    )
    axis.set_title(title)
    axis.set_xlabel("Конечное положение")
    axis.set_ylabel("Вероятность / плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def simulate_barrier_hitting_outcomes_for_random_walk(
    number_of_paths: int,
    upper_barrier: int,
    lower_barrier: int,
    probability_of_upward_step: float,
    maximum_number_of_steps: int,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Simulate which barrier is hit first and when it happens."""

    if lower_barrier >= 0 or upper_barrier <= 0:
        raise ValueError("lower_barrier must be negative and upper_barrier must be positive.")

    rng = create_random_number_generator(random_seed)
    terminal_labels = np.zeros(number_of_paths, dtype=int)
    first_hitting_times = np.full(number_of_paths, np.nan, dtype=float)

    for path_index in range(number_of_paths):
        current_position = 0
        for step_number in range(1, maximum_number_of_steps + 1):
            current_position += 1 if rng.random() < probability_of_upward_step else -1
            if current_position >= upper_barrier:
                terminal_labels[path_index] = 1
                first_hitting_times[path_index] = step_number
                break
            if current_position <= lower_barrier:
                terminal_labels[path_index] = -1
                first_hitting_times[path_index] = step_number
                break

    hit_mask = terminal_labels != 0
    return {
        "terminal_labels": terminal_labels,
        "first_hitting_times": first_hitting_times,
        "fraction_that_hit_any_barrier": hit_mask.mean(),
        "fraction_that_hit_upper_barrier_first": (terminal_labels == 1).mean(),
        "fraction_that_hit_lower_barrier_first": (terminal_labels == -1).mean(),
        "mean_hitting_time_among_hits": np.nanmean(first_hitting_times),
    }


def estimate_first_return_times_to_zero_for_symmetric_random_walk(
    number_of_paths: int,
    number_of_steps: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Estimate first positive return times to zero for a symmetric random walk."""

    _, simulated_paths = simulate_multiple_random_walk_paths(
        number_of_paths=number_of_paths,
        number_of_steps=number_of_steps,
        probability_of_upward_step=0.5,
        random_seed=random_seed,
    )
    first_return_times = np.full(number_of_paths, np.nan, dtype=float)
    return_to_zero_mask = simulated_paths[:, 1:] == 0.0
    for path_index, path_returns in enumerate(return_to_zero_mask):
        if path_returns.any():
            first_return_times[path_index] = int(np.argmax(path_returns) + 1)
    return first_return_times


def plot_first_return_time_histogram(
    first_return_times: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot first positive return times to zero."""

    valid_return_times = first_return_times[~np.isnan(first_return_times)]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(valid_return_times, bins=35, color="#72b7b2", alpha=0.75, edgecolor="white")
    axis.set_title(title)
    axis.set_xlabel("Первое положительное время возврата в 0")
    axis.set_ylabel("Число траекторий")
    axis.grid(alpha=0.3)
    return figure, axis


def calculate_theoretical_probability_of_hitting_upper_barrier_before_lower_barrier(
    upper_barrier: int,
    lower_barrier: int,
    probability_of_upward_step: float,
) -> float:
    """Return the gambler's ruin probability of hitting the upper barrier first."""

    lower_distance = abs(lower_barrier)
    total_width = upper_barrier + lower_distance
    if probability_of_upward_step == 0.5:
        return lower_distance / total_width

    ratio = (1.0 - probability_of_upward_step) / probability_of_upward_step
    numerator = 1.0 - ratio**lower_distance
    denominator = 1.0 - ratio**total_width
    return numerator / denominator


def calculate_theoretical_expected_hitting_time_for_symmetric_random_walk(
    upper_barrier: int,
    lower_barrier: int,
) -> float:
    """Return the expected time to hit either barrier for a symmetric walk."""

    return abs(lower_barrier) * upper_barrier


def plot_first_hitting_time_histogram(
    first_hitting_times: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot the distribution of first hitting times among successful hits."""

    valid_hitting_times = first_hitting_times[~np.isnan(first_hitting_times)]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(valid_hitting_times, bins=25, color="#e45756", alpha=0.75, edgecolor="white")
    axis.set_title(title)
    axis.set_xlabel("Число шагов до первого достижения барьера")
    axis.set_ylabel("Число траекторий")
    axis.grid(alpha=0.3)
    return figure, axis


def simulate_scaled_symmetric_random_walk_paths_for_brownian_approximation(
    number_of_paths: int,
    number_of_steps: int,
    final_time: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the diffusive scaling of a symmetric random walk."""

    time_grid = np.linspace(0.0, final_time, number_of_steps + 1)
    time_step = final_time / number_of_steps
    scaled_step_size = np.sqrt(time_step)
    _, scaled_paths = simulate_multiple_random_walk_paths(
        number_of_paths=number_of_paths,
        number_of_steps=number_of_steps,
        probability_of_upward_step=0.5,
        upward_step_size=scaled_step_size,
        downward_step_size=scaled_step_size,
        initial_position=0.0,
        random_seed=random_seed,
    )
    return time_grid, scaled_paths


def plot_scaled_random_walk_paths_with_brownian_variance_band(
    time_grid: np.ndarray,
    scaled_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot scaled random walk paths together with the Brownian one-sigma band."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for path_number in range(min(number_of_paths_to_plot, scaled_paths.shape[0])):
        axis.plot(
            time_grid,
            scaled_paths[path_number],
            linewidth=1.1,
            alpha=0.8,
        )
    standard_deviation_band = np.sqrt(time_grid)
    axis.plot(time_grid, standard_deviation_band, linestyle="--", color="black", label=r"$+\sqrt{t}$")
    axis.plot(time_grid, -standard_deviation_band, linestyle="--", color="black", label=r"$-\sqrt{t}$")
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Масштабированное положение")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
