from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_symmetric_random_walk_paths(
    number_of_paths: int,
    number_of_steps: int,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate symmetric random walk paths."""

    rng = create_random_number_generator(random_seed)
    increments = rng.choice([-1, 1], size=(number_of_paths, number_of_steps))
    paths = np.zeros((number_of_paths, number_of_steps + 1), dtype=int)
    paths[:, 1:] = np.cumsum(increments, axis=1)
    return np.arange(number_of_steps + 1), paths


def stop_random_walk_at_barriers(
    paths: np.ndarray,
    upper_barrier: int,
    lower_barrier: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stop random walk paths when they first hit either barrier."""

    stopped_values = np.empty(paths.shape[0], dtype=float)
    stopping_times = np.empty(paths.shape[0], dtype=int)
    stopped_paths = paths.copy().astype(float)
    for path_index, path in enumerate(paths):
        hit_indices = np.where((path >= upper_barrier) | (path <= lower_barrier))[0]
        stopping_time = int(hit_indices[0]) if hit_indices.size else paths.shape[1] - 1
        stopping_times[path_index] = stopping_time
        stopped_values[path_index] = path[stopping_time]
        stopped_paths[path_index, stopping_time:] = path[stopping_time]
    return stopped_paths, stopping_times, stopped_values


def calculate_theoretical_upper_barrier_probability_symmetric_walk(
    upper_barrier: int,
    lower_barrier: int,
) -> float:
    """Return P(hit upper first) for symmetric random walk."""

    return abs(lower_barrier) / (upper_barrier + abs(lower_barrier))


def calculate_theoretical_expected_stopping_time_symmetric_walk(
    upper_barrier: int,
    lower_barrier: int,
) -> float:
    """Return expected stopping time for symmetric random walk between barriers."""

    return float(upper_barrier * abs(lower_barrier))


def simulate_doubling_strategy_until_win_or_limit(
    number_of_trials: int,
    maximum_number_of_losses: int,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a martingale betting strategy with a finite loss limit."""

    rng = create_random_number_generator(random_seed)
    profits = np.empty(number_of_trials, dtype=float)
    maximum_bets = np.empty(number_of_trials, dtype=float)
    for trial in range(number_of_trials):
        stake = 1.0
        wealth_change = 0.0
        maximum_bet = stake
        for _ in range(maximum_number_of_losses + 1):
            maximum_bet = max(maximum_bet, stake)
            if rng.random() < 0.5:
                wealth_change += stake
                break
            wealth_change -= stake
            stake *= 2.0
        profits[trial] = wealth_change
        maximum_bets[trial] = maximum_bet
    return profits, maximum_bets


def plot_random_walk_and_stopped_paths(
    step_grid: np.ndarray,
    paths: np.ndarray,
    stopped_paths: np.ndarray,
    upper_barrier: int,
    lower_barrier: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot raw and stopped random walk paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(8, paths.shape[0])):
        axis.plot(step_grid, paths[index], alpha=0.25, linewidth=1.0, color="#4c78a8")
        axis.plot(step_grid, stopped_paths[index], alpha=0.9, linewidth=1.4)
    axis.axhline(upper_barrier, color="#54a24b", linestyle="--", label="Верхний барьер")
    axis.axhline(lower_barrier, color="#e45756", linestyle="--", label="Нижний барьер")
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("S_n")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_stopping_time_distribution(stopping_times: np.ndarray, title: str) -> tuple[Figure, Axes]:
    """Plot stopping time distribution."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(stopping_times, bins=35, color="#f58518", alpha=0.75, edgecolor="white")
    axis.set_title(title)
    axis.set_xlabel("Время остановки")
    axis.set_ylabel("Число траекторий")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_martingale_mean_over_time(
    step_grid: np.ndarray,
    paths: np.ndarray,
    stopped_paths: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical means of original and stopped martingales."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(step_grid, paths.mean(axis=0), label="E[S_n] по симуляции", linewidth=2.0)
    axis.plot(step_grid, stopped_paths.mean(axis=0), label="E[S_{n∧τ}] по симуляции", linewidth=2.0)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Среднее")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
