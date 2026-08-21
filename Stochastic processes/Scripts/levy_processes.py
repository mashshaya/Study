from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_brownian_motion_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    volatility: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Brownian Levy process paths."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    dt = final_time / number_of_time_steps
    increments = volatility * np.sqrt(dt) * rng.normal(size=(number_of_paths, number_of_time_steps))
    paths = np.zeros((number_of_paths, number_of_time_steps + 1))
    paths[:, 1:] = np.cumsum(increments, axis=1)
    return time_grid, paths


def simulate_poisson_process_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    intensity: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Poisson Levy process paths."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    dt = final_time / number_of_time_steps
    increments = rng.poisson(intensity * dt, size=(number_of_paths, number_of_time_steps))
    paths = np.zeros((number_of_paths, number_of_time_steps + 1))
    paths[:, 1:] = np.cumsum(increments, axis=1)
    return time_grid, paths


def simulate_variance_gamma_terminal_values(
    number_of_paths: int,
    final_time: float,
    theta: float,
    sigma: float,
    nu: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal values of a variance-gamma process."""

    rng = create_random_number_generator(random_seed)
    gamma_time = rng.gamma(shape=final_time / nu, scale=nu, size=number_of_paths)
    return theta * gamma_time + sigma * np.sqrt(gamma_time) * rng.normal(size=number_of_paths)


def calculate_empirical_characteristic_function(
    samples: np.ndarray,
    u_values: np.ndarray,
) -> np.ndarray:
    """Calculate empirical characteristic function E exp(i u X)."""

    return np.array([np.mean(np.exp(1j * u * samples)) for u in u_values])


def calculate_brownian_characteristic_function(
    u_values: np.ndarray,
    final_time: float,
    volatility: float,
) -> np.ndarray:
    """Return Brownian characteristic function."""

    return np.exp(-0.5 * volatility**2 * u_values**2 * final_time)


def calculate_compound_poisson_normal_jump_characteristic_function(
    u_values: np.ndarray,
    final_time: float,
    intensity: float,
    jump_mean: float,
    jump_std: float,
) -> np.ndarray:
    """Return characteristic function of compound Poisson with normal jumps."""

    jump_cf = np.exp(1j * u_values * jump_mean - 0.5 * jump_std**2 * u_values**2)
    return np.exp(final_time * intensity * (jump_cf - 1.0))


def simulate_compound_poisson_normal_terminal_values(
    number_of_paths: int,
    final_time: float,
    intensity: float,
    jump_mean: float,
    jump_std: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal compound Poisson values with normal jumps."""

    rng = create_random_number_generator(random_seed)
    counts = rng.poisson(intensity * final_time, size=number_of_paths)
    values = np.zeros(number_of_paths)
    for index, count in enumerate(counts):
        if count:
            values[index] = rng.normal(jump_mean, jump_std, size=count).sum()
    return values


def simulate_compound_poisson_normal_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    final_time: float,
    intensity: float,
    jump_mean: float,
    jump_std: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate compound Poisson Levy paths with normal jumps."""

    rng = create_random_number_generator(random_seed)
    time_grid = np.linspace(0.0, final_time, number_of_time_steps + 1)
    dt = final_time / number_of_time_steps
    paths = np.zeros((number_of_paths, number_of_time_steps + 1))
    for step in range(number_of_time_steps):
        counts = rng.poisson(intensity * dt, size=number_of_paths)
        jump_sums = np.zeros(number_of_paths)
        for path_index, count in enumerate(counts):
            if count:
                jump_sums[path_index] = rng.normal(jump_mean, jump_std, size=count).sum()
        paths[:, step + 1] = paths[:, step] + jump_sums
    return time_grid, paths


def plot_levy_process_paths(
    time_grid: np.ndarray,
    paths_by_name: dict[str, np.ndarray],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot representative paths of several Levy processes."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, paths in paths_by_name.items():
        axis.plot(time_grid, paths[0], linewidth=1.5, label=name)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("X(t)")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_characteristic_function_comparison(
    u_values: np.ndarray,
    empirical_cf: np.ndarray,
    theoretical_cf: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot real parts of empirical and theoretical characteristic functions."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(u_values, empirical_cf.real, label="Re empirical", linewidth=2.0)
    axis.plot(u_values, theoretical_cf.real, "--", label="Re theory", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel("u")
    axis.set_ylabel("Re phi(u)")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
