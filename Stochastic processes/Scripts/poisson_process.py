from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import expon, poisson


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_multiple_homogeneous_poisson_process_paths_by_increments(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    intensity: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate homogeneous Poisson process paths on an equally spaced grid."""

    if number_of_paths <= 0 or number_of_time_steps <= 0:
        raise ValueError("number_of_paths and number_of_time_steps must be positive.")
    if final_time <= 0.0 or intensity < 0.0:
        raise ValueError("final_time must be positive and intensity must be non-negative.")

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    increments = rng.poisson(
        lam=intensity * time_step,
        size=(number_of_paths, number_of_time_steps),
    )
    paths = np.zeros((number_of_paths, number_of_time_steps + 1), dtype=int)
    paths[:, 1:] = np.cumsum(increments, axis=1)
    return time_grid, paths


def simulate_homogeneous_poisson_process_event_times(
    final_time: float,
    intensity: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate event times through exponential interarrival times."""

    if final_time <= 0.0 or intensity <= 0.0:
        raise ValueError("final_time and intensity must be positive.")

    rng = create_random_number_generator(random_seed)
    event_times: list[float] = []
    current_time = 0.0
    while True:
        current_time += float(rng.exponential(scale=1.0 / intensity))
        if current_time > final_time:
            break
        event_times.append(current_time)
    return np.array(event_times)


def simulate_exponential_waiting_times(
    number_of_waiting_times: int,
    intensity: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate exponential waiting times between Poisson events."""

    if number_of_waiting_times <= 0 or intensity <= 0.0:
        raise ValueError("number_of_waiting_times and intensity must be positive.")
    rng = create_random_number_generator(random_seed)
    return rng.exponential(scale=1.0 / intensity, size=number_of_waiting_times)


def calculate_empirical_mean_and_variance_across_poisson_paths(
    poisson_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical mean and variance of N(t) across paths."""

    return poisson_paths.mean(axis=0), poisson_paths.var(axis=0)


def calculate_theoretical_mean_and_variance_for_homogeneous_poisson_process(
    time_grid: np.ndarray,
    intensity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical mean and variance lambda t for a Poisson process."""

    theoretical_values = intensity * time_grid
    return theoretical_values, theoretical_values.copy()


def simulate_nonhomogeneous_poisson_process_paths_by_thinning(
    number_of_paths: int,
    final_time: float,
    intensity_function: Callable[[np.ndarray], np.ndarray],
    maximum_intensity: float,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """Simulate nonhomogeneous Poisson event times by thinning."""

    if number_of_paths <= 0 or final_time <= 0.0 or maximum_intensity <= 0.0:
        raise ValueError("number_of_paths, final_time, and maximum_intensity must be positive.")

    rng = create_random_number_generator(random_seed)
    simulated_event_times: list[np.ndarray] = []
    for _ in range(number_of_paths):
        candidate_count = rng.poisson(maximum_intensity * final_time)
        candidate_times = np.sort(rng.uniform(0.0, final_time, size=candidate_count))
        acceptance_probabilities = intensity_function(candidate_times) / maximum_intensity
        accepted = rng.random(candidate_count) <= acceptance_probabilities
        simulated_event_times.append(candidate_times[accepted])
    return simulated_event_times


def convert_event_times_to_count_paths(
    event_times_by_path: list[np.ndarray],
    time_grid: np.ndarray,
) -> np.ndarray:
    """Convert event-time representation to count paths on a grid."""

    paths = np.zeros((len(event_times_by_path), len(time_grid)), dtype=int)
    for path_index, event_times in enumerate(event_times_by_path):
        paths[path_index] = np.searchsorted(event_times, time_grid, side="right")
    return paths


def calculate_cumulative_intensity_for_sinusoidal_poisson_process(
    time_grid: np.ndarray,
    base_intensity: float,
    amplitude: float,
    period: float,
) -> np.ndarray:
    """Return integral of base + amplitude*sin(2*pi*t/period)."""

    return (
        base_intensity * time_grid
        + amplitude * period / (2.0 * np.pi) * (1.0 - np.cos(2.0 * np.pi * time_grid / period))
    )


def simulate_superposed_poisson_terminal_counts(
    number_of_replications: int,
    final_time: float,
    intensities: list[float],
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal counts of independent superposed Poisson processes."""

    rng = create_random_number_generator(random_seed)
    component_counts = [
        rng.poisson(lam=intensity * final_time, size=number_of_replications)
        for intensity in intensities
    ]
    return np.sum(component_counts, axis=0)


def simulate_thinned_poisson_terminal_counts(
    number_of_replications: int,
    final_time: float,
    original_intensity: float,
    retention_probability: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate retained and removed counts after thinning a Poisson process."""

    if not 0.0 <= retention_probability <= 1.0:
        raise ValueError("retention_probability must lie in [0, 1].")

    rng = create_random_number_generator(random_seed)
    total_counts = rng.poisson(original_intensity * final_time, size=number_of_replications)
    retained_counts = rng.binomial(total_counts, retention_probability)
    removed_counts = total_counts - retained_counts
    return retained_counts, removed_counts


def plot_poisson_process_sample_paths(
    time_grid: np.ndarray,
    poisson_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot step paths of a Poisson process."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for path_index in range(min(number_of_paths_to_plot, poisson_paths.shape[0])):
        axis.step(
            time_grid,
            poisson_paths[path_index],
            where="post",
            linewidth=1.4,
            alpha=0.85,
            label=f"Траектория {path_index + 1}" if path_index < 6 else None,
        )
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Число событий N(t)")
    if min(number_of_paths_to_plot, poisson_paths.shape[0]) <= 6:
        axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_poisson_empirical_and_theoretical_mean_variance(
    time_grid: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_variance: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_variance: np.ndarray,
    title_prefix: str,
) -> tuple[Figure, np.ndarray]:
    """Plot empirical and theoretical mean and variance for N(t)."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_grid, empirical_mean, label="Эмпирическое среднее", linewidth=2.0)
    axes[0].plot(time_grid, theoretical_mean, "--", label="Теория", linewidth=2.0)
    axes[0].set_title(f"{title_prefix}: среднее")
    axes[0].set_xlabel("Время")
    axes[0].set_ylabel("E[N(t)]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_grid, empirical_variance, label="Эмпирическая дисперсия", linewidth=2.0)
    axes[1].plot(time_grid, theoretical_variance, "--", label="Теория", linewidth=2.0)
    axes[1].set_title(f"{title_prefix}: дисперсия")
    axes[1].set_xlabel("Время")
    axes[1].set_ylabel("Var[N(t)]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


def plot_poisson_terminal_distribution_with_theory(
    terminal_counts: np.ndarray,
    theoretical_mean: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical terminal count distribution against Poisson probabilities."""

    figure, axis = plt.subplots(figsize=(10, 5))
    max_count = int(max(terminal_counts.max(), poisson.ppf(0.999, theoretical_mean)))
    bins = np.arange(-0.5, max_count + 1.5)
    axis.hist(
        terminal_counts,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#4c78a8",
        label="Симуляция",
    )
    count_values = np.arange(0, max_count + 1)
    axis.plot(
        count_values,
        poisson.pmf(count_values, theoretical_mean),
        "o-",
        color="#f58518",
        linewidth=2.0,
        label="Точная вероятность",
    )
    axis.set_title(title)
    axis.set_xlabel("N(T)")
    axis.set_ylabel("Вероятность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_exponential_waiting_time_distribution(
    waiting_times: np.ndarray,
    intensity: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot exponential waiting times and theoretical density."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(waiting_times, bins=45, density=True, alpha=0.55, color="#54a24b", label="Симуляция")
    dense_grid = np.linspace(0.0, np.quantile(waiting_times, 0.995), 500)
    axis.plot(
        dense_grid,
        expon.pdf(dense_grid, scale=1.0 / intensity),
        color="#e45756",
        linewidth=2.2,
        label="Экспоненциальная плотность",
    )
    axis.set_title(title)
    axis.set_xlabel("Время ожидания")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_nonhomogeneous_poisson_intensity_and_mean(
    time_grid: np.ndarray,
    intensity_values: np.ndarray,
    empirical_mean: np.ndarray,
    theoretical_mean: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot nonhomogeneous intensity and cumulative mean."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_grid, intensity_values, color="#4c78a8", linewidth=2.0)
    axes[0].set_title("Мгновенная интенсивность")
    axes[0].set_xlabel("Время")
    axes[0].set_ylabel("lambda(t)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time_grid, empirical_mean, label="Эмпирическое среднее", linewidth=2.0)
    axes[1].plot(time_grid, theoretical_mean, "--", label="Интеграл интенсивности", linewidth=2.0)
    axes[1].set_title(title)
    axes[1].set_xlabel("Время")
    axes[1].set_ylabel("E[N(t)]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes
