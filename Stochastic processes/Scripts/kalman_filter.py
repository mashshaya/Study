from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def simulate_local_level_state_space_model(
    number_of_steps: int,
    initial_state: float,
    transition_coefficient: float,
    process_noise_std: float,
    observation_noise_std: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a scalar linear Gaussian state-space model."""

    rng = create_random_number_generator(random_seed)
    states = np.empty(number_of_steps)
    observations = np.empty(number_of_steps)
    states[0] = initial_state
    observations[0] = states[0] + observation_noise_std * rng.normal()
    for step in range(1, number_of_steps):
        states[step] = transition_coefficient * states[step - 1] + process_noise_std * rng.normal()
        observations[step] = states[step] + observation_noise_std * rng.normal()
    return states, observations


def apply_scalar_kalman_filter(
    observations: np.ndarray,
    transition_coefficient: float,
    observation_coefficient: float,
    process_noise_variance: float,
    observation_noise_variance: float,
    initial_state_estimate: float,
    initial_estimate_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply scalar Kalman filter and return estimates, variances, gains."""

    estimates = np.empty_like(observations, dtype=float)
    variances = np.empty_like(observations, dtype=float)
    gains = np.empty_like(observations, dtype=float)
    previous_estimate = initial_state_estimate
    previous_variance = initial_estimate_variance
    for step, observation in enumerate(observations):
        predicted_estimate = transition_coefficient * previous_estimate
        predicted_variance = transition_coefficient**2 * previous_variance + process_noise_variance
        innovation_variance = observation_coefficient**2 * predicted_variance + observation_noise_variance
        kalman_gain = predicted_variance * observation_coefficient / innovation_variance
        estimates[step] = predicted_estimate + kalman_gain * (observation - observation_coefficient * predicted_estimate)
        variances[step] = (1.0 - kalman_gain * observation_coefficient) * predicted_variance
        gains[step] = kalman_gain
        previous_estimate = estimates[step]
        previous_variance = variances[step]
    return estimates, variances, gains


def calculate_root_mean_squared_error(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Calculate RMSE."""

    return float(np.sqrt(np.mean((truth - estimate) ** 2)))


def plot_kalman_filter_results(
    states: np.ndarray,
    observations: np.ndarray,
    estimates: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot true state, noisy observations, and filtered estimates."""

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(states, label="Истинное состояние", linewidth=2.0)
    axis.scatter(np.arange(observations.size), observations, s=14, alpha=0.45, label="Наблюдения")
    axis.plot(estimates, label="Оценка Калмана", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Значение")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis


def plot_kalman_gain_and_uncertainty(
    gains: np.ndarray,
    variances: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot Kalman gain and posterior uncertainty."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(gains, linewidth=2.0)
    axes[0].set_title("Коэффициент Калмана")
    axes[0].grid(alpha=0.3)
    axes[1].plot(variances, linewidth=2.0)
    axes[1].set_title("Дисперсия ошибки оценки")
    axes[1].grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes
