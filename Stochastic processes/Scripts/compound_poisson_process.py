from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_multiple_compound_poisson_process_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    intensity: float,
    jump_mean: float,
    jump_standard_deviation: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate compound Poisson paths with normally distributed jump sizes."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    time_step = final_time / number_of_time_steps
    paths = np.zeros((number_of_paths, number_of_time_steps + 1), dtype=float)
    for step in range(number_of_time_steps):
        jump_counts = rng.poisson(intensity * time_step, size=number_of_paths)
        jump_sums = np.zeros(number_of_paths, dtype=float)
        for path_index, count in enumerate(jump_counts):
            if count:
                jump_sums[path_index] = rng.normal(jump_mean, jump_standard_deviation, size=count).sum()
        paths[:, step + 1] = paths[:, step] + jump_sums
    return time_grid, paths


def simulate_compound_poisson_terminal_values(
    number_of_replications: int,
    final_time: float,
    intensity: float,
    jump_sampler,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal compound Poisson values for a generic jump sampler."""

    rng = create_random_number_generator(random_seed)
    counts = rng.poisson(intensity * final_time, size=number_of_replications)
    totals = np.zeros(number_of_replications, dtype=float)
    for index, count in enumerate(counts):
        if count:
            totals[index] = jump_sampler(rng, int(count)).sum()
    return totals


def calculate_theoretical_compound_poisson_mean_and_variance(
    time_grid: np.ndarray,
    intensity: float,
    jump_mean: float,
    jump_second_moment: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E[S_t] and Var(S_t) for a compound Poisson process."""

    mean = intensity * time_grid * jump_mean
    variance = intensity * time_grid * jump_second_moment
    return mean, variance


def calculate_empirical_mean_and_variance(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical mean and variance across paths."""

    return paths.mean(axis=0), paths.var(axis=0)


def plot_compound_poisson_paths(
    time_grid: np.ndarray,
    paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot compound Poisson sample paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(number_of_paths_to_plot, paths.shape[0])):
        axis.step(time_grid, paths[index], where="post", linewidth=1.3, alpha=0.85)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Накопленная сумма")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_compound_poisson_mean_variance(
    time_grid: np.ndarray,
    empirical_mean: np.ndarray,
    empirical_variance: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_variance: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot empirical and theoretical mean and variance."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time_grid, empirical_mean, label="Симуляция", linewidth=2.0)
    axes[0].plot(time_grid, theoretical_mean, "--", label="Теория", linewidth=2.0)
    axes[0].set_title(f"{title}: среднее")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_grid, empirical_variance, label="Симуляция", linewidth=2.0)
    axes[1].plot(time_grid, theoretical_variance, "--", label="Теория", linewidth=2.0)
    axes[1].set_title(f"{title}: дисперсия")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


def plot_terminal_compound_poisson_distribution(
    terminal_values: np.ndarray,
    theoretical_mean: float,
    theoretical_variance: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot terminal values with a normal approximation."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(terminal_values, bins=60, density=True, alpha=0.55, color="#4c78a8", label="Симуляция")
    grid = np.linspace(np.quantile(terminal_values, 0.001), np.quantile(terminal_values, 0.999), 700)
    axis.plot(grid, norm.pdf(grid, theoretical_mean, np.sqrt(theoretical_variance)), color="#f58518", linewidth=2.2, label="Нормальная аппроксимация")
    axis.set_title(title)
    axis.set_xlabel("S(T)")
    axis.set_ylabel("Плотность")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
