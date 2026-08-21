from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_sde_paths_with_euler_maruyama(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    initial_value: float,
    drift_function,
    diffusion_function,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a one-dimensional SDE using Euler-Maruyama."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    dt = final_time / number_of_time_steps
    brownian_increments = np.sqrt(dt) * rng.normal(size=(number_of_paths, number_of_time_steps))
    paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for step in range(number_of_time_steps):
        current_time = time_grid[step]
        current_value = paths[:, step]
        paths[:, step + 1] = (
            current_value
            + drift_function(current_time, current_value) * dt
            + diffusion_function(current_time, current_value) * brownian_increments[:, step]
        )
    return time_grid, paths, brownian_increments


def calculate_exact_gbm_paths_from_brownian_increments(
    brownian_increments: np.ndarray,
    initial_value: float,
    drift: float,
    volatility: float,
    final_time: float,
) -> np.ndarray:
    """Calculate exact GBM paths using the same Brownian increments."""

    number_of_time_steps = brownian_increments.shape[1]
    dt = final_time / number_of_time_steps
    log_increments = (drift - 0.5 * volatility**2) * dt + volatility * brownian_increments
    log_paths = np.empty((brownian_increments.shape[0], number_of_time_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(initial_value)
    log_paths[:, 1:] = np.log(initial_value) + np.cumsum(log_increments, axis=1)
    return np.exp(log_paths)


def calculate_exact_ou_paths_from_brownian_increments_approximately(
    brownian_increments: np.ndarray,
    initial_value: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    final_time: float,
) -> np.ndarray:
    """Calculate fine-grid OU reference paths with exact transition distribution."""

    number_of_paths, number_of_time_steps = brownian_increments.shape
    dt = final_time / number_of_time_steps
    decay = np.exp(-mean_reversion_speed * dt)
    innovation_std = volatility * np.sqrt((1.0 - decay**2) / (2.0 * mean_reversion_speed))
    normalized_shocks = brownian_increments / np.sqrt(dt)
    paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for step in range(number_of_time_steps):
        paths[:, step + 1] = (
            long_term_mean
            + (paths[:, step] - long_term_mean) * decay
            + innovation_std * normalized_shocks[:, step]
        )
    return paths


def estimate_terminal_strong_errors_for_gbm(
    step_counts: list[int],
    number_of_paths: int,
    final_time: float,
    initial_value: float,
    drift: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate strong and weak terminal errors for GBM Euler-Maruyama."""

    strong_errors = []
    weak_errors = []
    for index, steps in enumerate(step_counts):
        _, em_paths, increments = simulate_sde_paths_with_euler_maruyama(
            number_of_paths,
            steps,
            final_time,
            initial_value,
            lambda _t, x: drift * x,
            lambda _t, x: volatility * x,
            None if random_seed is None else random_seed + index,
        )
        exact_paths = calculate_exact_gbm_paths_from_brownian_increments(
            increments,
            initial_value,
            drift,
            volatility,
            final_time,
        )
        strong_errors.append(np.mean(np.abs(em_paths[:, -1] - exact_paths[:, -1])))
        weak_errors.append(abs(em_paths[:, -1].mean() - exact_paths[:, -1].mean()))
    return np.array(step_counts), np.array(strong_errors), np.array(weak_errors)


def plot_sde_paths(
    time_grid: np.ndarray,
    paths: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot sample SDE paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(12, paths.shape[0])):
        axis.plot(time_grid, paths[index], alpha=0.85, linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("X(t)")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_error_convergence(
    step_counts: np.ndarray,
    strong_errors: np.ndarray,
    weak_errors: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot strong and weak error convergence."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.loglog(step_counts, strong_errors, marker="o", label="Strong error")
    axis.loglog(step_counts, weak_errors, marker="o", label="Weak mean error")
    axis.set_title(title)
    axis.set_xlabel("Число шагов")
    axis.set_ylabel("Ошибка")
    axis.legend()
    axis.grid(alpha=0.3, which="both")
    return figure, axis
