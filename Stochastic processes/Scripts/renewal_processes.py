from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_renewal_event_times(
    final_time: float,
    interarrival_sampler,
    random_number_generator: np.random.Generator,
) -> np.ndarray:
    """Simulate renewal event times up to final_time."""

    times: list[float] = []
    current_time = 0.0
    while True:
        current_time += float(interarrival_sampler(random_number_generator, 1)[0])
        if current_time > final_time:
            break
        times.append(current_time)
    return np.array(times)


def simulate_multiple_renewal_count_paths(
    number_of_paths: int,
    time_grid: np.ndarray,
    interarrival_sampler,
    random_seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Simulate renewal event times and count paths."""

    rng = create_random_number_generator(random_seed)
    event_times_by_path = [
        simulate_renewal_event_times(float(time_grid[-1]), interarrival_sampler, rng)
        for _ in range(number_of_paths)
    ]
    count_paths = np.zeros((number_of_paths, time_grid.size), dtype=int)
    for index, event_times in enumerate(event_times_by_path):
        count_paths[index] = np.searchsorted(event_times, time_grid, side="right")
    return event_times_by_path, count_paths


def calculate_empirical_renewal_mean_and_variance(count_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate empirical renewal count mean and variance."""

    return count_paths.mean(axis=0), count_paths.var(axis=0)


def calculate_age_and_residual_life_at_time(event_times: np.ndarray, observation_time: float) -> tuple[float, float]:
    """Calculate age and residual life at a fixed observation time."""

    previous_events = event_times[event_times <= observation_time]
    next_events = event_times[event_times > observation_time]
    age = observation_time - previous_events[-1] if previous_events.size else observation_time
    residual = next_events[0] - observation_time if next_events.size else np.nan
    return float(age), float(residual)


def calculate_age_and_residual_samples(
    event_times_by_path: list[np.ndarray],
    observation_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate age and residual life samples across renewal paths."""

    ages = []
    residuals = []
    for event_times in event_times_by_path:
        age, residual = calculate_age_and_residual_life_at_time(event_times, observation_time)
        ages.append(age)
        residuals.append(residual)
    return np.array(ages), np.array(residuals)


def plot_renewal_count_paths(
    time_grid: np.ndarray,
    count_paths: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot renewal count paths."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for index in range(min(number_of_paths_to_plot, count_paths.shape[0])):
        axis.step(time_grid, count_paths[index], where="post", linewidth=1.3, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("N(t)")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_renewal_mean_functions(
    time_grid: np.ndarray,
    mean_functions_by_name: dict[str, np.ndarray],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical renewal functions for several interarrival laws."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, mean_values in mean_functions_by_name.items():
        axis.plot(time_grid, mean_values, linewidth=2.0, label=name)
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("E[N(t)]")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_age_and_residual_distributions(
    ages: np.ndarray,
    residuals: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot age and residual life histograms."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(ages, bins=45, color="#4c78a8", alpha=0.75, edgecolor="white")
    axes[0].set_title("Возраст")
    axes[0].set_xlabel("Время с последнего восстановления")
    axes[0].grid(alpha=0.3)
    valid_residuals = residuals[~np.isnan(residuals)]
    axes[1].hist(valid_residuals, bins=45, color="#f58518", alpha=0.75, edgecolor="white")
    axes[1].set_title("Остаточное время")
    axes[1].set_xlabel("Время до следующего восстановления")
    axes[1].grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes
