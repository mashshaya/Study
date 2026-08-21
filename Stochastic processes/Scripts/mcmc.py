from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.special import expit
from scipy.stats import norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def unnormalized_mixture_log_density(x: np.ndarray | float) -> np.ndarray | float:
    """Log density of a bimodal normal mixture up to a constant."""

    x_array = np.asarray(x)
    density = 0.45 * norm.pdf(x_array, loc=-2.0, scale=0.7) + 0.55 * norm.pdf(x_array, loc=2.0, scale=1.0)
    return np.log(density)


def run_random_walk_metropolis_sampler(
    log_density_function,
    initial_value: float,
    number_of_iterations: int,
    proposal_standard_deviation: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Run random-walk Metropolis sampling."""

    rng = create_random_number_generator(random_seed)
    samples = np.empty(number_of_iterations)
    current = initial_value
    current_log_density = float(log_density_function(current))
    accepted = 0
    for iteration in range(number_of_iterations):
        proposal = current + proposal_standard_deviation * rng.normal()
        proposal_log_density = float(log_density_function(proposal))
        if np.log(rng.random()) < proposal_log_density - current_log_density:
            current = proposal
            current_log_density = proposal_log_density
            accepted += 1
        samples[iteration] = current
    return samples, accepted / number_of_iterations


def calculate_autocorrelation(samples: np.ndarray, max_lag: int) -> np.ndarray:
    """Calculate sample autocorrelation up to max_lag."""

    centered = samples - samples.mean()
    denominator = np.dot(centered, centered)
    return np.array([1.0] + [np.dot(centered[:-lag], centered[lag:]) / denominator for lag in range(1, max_lag + 1)])


def estimate_effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> float:
    """Estimate effective sample size using positive autocorrelation sum."""

    autocorrelations = calculate_autocorrelation(samples, max_lag)
    positive_tail = autocorrelations[1:][autocorrelations[1:] > 0.0]
    return float(samples.size / (1.0 + 2.0 * positive_tail.sum()))


def generate_logistic_regression_data(
    number_of_observations: int,
    true_intercept: float,
    true_slope: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple Bayesian logistic regression data."""

    rng = create_random_number_generator(random_seed)
    x = rng.normal(size=number_of_observations)
    probabilities = expit(true_intercept + true_slope * x)
    y = rng.binomial(1, probabilities)
    return x, y


def make_logistic_regression_log_posterior(x: np.ndarray, y: np.ndarray, prior_std: float):
    """Create log posterior for slope with fixed zero intercept."""

    def log_posterior(beta: float) -> float:
        logits = beta * x
        log_likelihood = np.sum(y * logits - np.logaddexp(0.0, logits))
        log_prior = -0.5 * (beta / prior_std) ** 2
        return float(log_likelihood + log_prior)

    return log_posterior


def plot_mcmc_trace_and_histogram(
    samples: np.ndarray,
    burn_in: int,
    grid: np.ndarray,
    target_density: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot MCMC trace and posterior histogram."""

    post_burn = samples[burn_in:]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(samples, linewidth=0.8)
    axes[0].axvline(burn_in, color="black", linestyle="--", label="burn-in")
    axes[0].set_title("Траектория цепи")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].hist(post_burn, bins=60, density=True, alpha=0.6, label="MCMC")
    axes[1].plot(grid, target_density, linewidth=2.2, label="Целевая плотность")
    axes[1].set_title("Гистограмма после burn-in")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes


def plot_autocorrelation(autocorrelations: np.ndarray, title: str) -> tuple[Figure, Axes]:
    """Plot autocorrelation function."""

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.stem(np.arange(autocorrelations.size), autocorrelations)
    axis.set_title(title)
    axis.set_xlabel("Lag")
    axis.set_ylabel("ACF")
    axis.grid(alpha=0.3)
    return figure, axis
