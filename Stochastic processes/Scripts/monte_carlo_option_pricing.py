from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def calculate_black_scholes_call_and_put_prices(
    spot_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
) -> tuple[float, float]:
    """Return Black-Scholes European call and put prices."""

    d1 = (
        np.log(spot_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    call = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * maturity) * norm.cdf(d2)
    put = strike_price * np.exp(-risk_free_rate * maturity) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    return float(call), float(put)


def simulate_risk_neutral_terminal_prices(
    number_of_paths: int,
    spot_price: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    random_seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate terminal GBM prices under the risk-neutral measure."""

    rng = create_random_number_generator(random_seed)
    if antithetic:
        half = int(np.ceil(number_of_paths / 2))
        z = rng.normal(size=half)
        z = np.concatenate([z, -z])[:number_of_paths]
    else:
        z = rng.normal(size=number_of_paths)
    return spot_price * np.exp(
        (risk_free_rate - 0.5 * volatility**2) * maturity
        + volatility * np.sqrt(maturity) * z
    )


def estimate_discounted_option_price_from_payoffs(
    payoffs: np.ndarray,
    risk_free_rate: float,
    maturity: float,
) -> tuple[float, float, tuple[float, float]]:
    """Estimate discounted option price, standard error, and 95% CI."""

    discounted = np.exp(-risk_free_rate * maturity) * payoffs
    estimate = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / np.sqrt(discounted.size))
    return estimate, standard_error, (estimate - 1.96 * standard_error, estimate + 1.96 * standard_error)


def calculate_european_call_put_payoffs(
    terminal_prices: np.ndarray,
    strike_price: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate European call and put payoffs."""

    return np.maximum(terminal_prices - strike_price, 0.0), np.maximum(strike_price - terminal_prices, 0.0)


def simulate_risk_neutral_price_paths(
    number_of_paths: int,
    number_of_time_steps: int,
    spot_price: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate risk-neutral GBM price paths."""

    rng = create_random_number_generator(random_seed)
    dt = maturity / number_of_time_steps
    time_grid = np.linspace(0.0, maturity, number_of_time_steps + 1)
    shocks = rng.normal(size=(number_of_paths, number_of_time_steps))
    log_increments = (risk_free_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_paths = np.empty((number_of_paths, number_of_time_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(spot_price)
    log_paths[:, 1:] = np.log(spot_price) + np.cumsum(log_increments, axis=1)
    return time_grid, np.exp(log_paths)


def calculate_arithmetic_asian_call_payoffs(
    price_paths: np.ndarray,
    strike_price: float,
) -> np.ndarray:
    """Calculate arithmetic-average Asian call payoffs."""

    average_prices = price_paths[:, 1:].mean(axis=1)
    return np.maximum(average_prices - strike_price, 0.0)


def calculate_up_and_out_call_payoffs(
    price_paths: np.ndarray,
    strike_price: float,
    barrier: float,
) -> np.ndarray:
    """Calculate up-and-out call payoffs on a discrete monitoring grid."""

    knocked_out = np.max(price_paths, axis=1) >= barrier
    vanilla_payoff = np.maximum(price_paths[:, -1] - strike_price, 0.0)
    return np.where(knocked_out, 0.0, vanilla_payoff)


def plot_monte_carlo_convergence(
    path_counts: np.ndarray,
    estimates: np.ndarray,
    true_price: float,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot Monte Carlo convergence against analytical price."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(path_counts, estimates, marker="o", linewidth=2.0, label="MC estimate")
    axis.axhline(true_price, color="black", linestyle="--", label="Black-Scholes")
    axis.set_xscale("log")
    axis.set_title(title)
    axis.set_xlabel("Число симуляций")
    axis.set_ylabel("Цена")
    axis.legend()
    axis.grid(alpha=0.3)
    return figure, axis
