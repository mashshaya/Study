from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_random_number_generator(random_seed: int | None) -> np.random.Generator:
    """Create a reproducible random number generator."""

    return np.random.default_rng(random_seed)


def validate_transition_matrix(transition_matrix: np.ndarray) -> None:
    """Validate that a transition matrix is stochastic."""

    if transition_matrix.ndim != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if np.any(transition_matrix < -1e-12):
        raise ValueError("transition probabilities must be non-negative.")
    if not np.allclose(transition_matrix.sum(axis=1), 1.0):
        raise ValueError("rows of transition_matrix must sum to 1.")


def simulate_multiple_markov_chain_paths(
    transition_matrix: np.ndarray,
    initial_state: int,
    number_of_steps: int,
    number_of_paths: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate multiple finite-state Markov chain paths."""

    validate_transition_matrix(transition_matrix)
    if number_of_steps <= 0 or number_of_paths <= 0:
        raise ValueError("number_of_steps and number_of_paths must be positive.")

    rng = create_random_number_generator(random_seed)
    number_of_states = transition_matrix.shape[0]
    paths = np.empty((number_of_paths, number_of_steps + 1), dtype=int)
    paths[:, 0] = initial_state
    for step in range(number_of_steps):
        for state in range(number_of_states):
            path_mask = paths[:, step] == state
            number_in_state = int(path_mask.sum())
            if number_in_state:
                paths[path_mask, step + 1] = rng.choice(
                    number_of_states,
                    size=number_in_state,
                    p=transition_matrix[state],
                )
    return paths


def calculate_state_frequencies_over_time(
    simulated_paths: np.ndarray,
    number_of_states: int,
) -> np.ndarray:
    """Calculate empirical state frequencies at each time step."""

    frequencies = np.zeros((simulated_paths.shape[1], number_of_states), dtype=float)
    for state in range(number_of_states):
        frequencies[:, state] = (simulated_paths == state).mean(axis=0)
    return frequencies


def calculate_distribution_after_each_step(
    initial_distribution: np.ndarray,
    transition_matrix: np.ndarray,
    number_of_steps: int,
) -> np.ndarray:
    """Calculate row distributions pi_0 P^n for n from 0 to number_of_steps."""

    validate_transition_matrix(transition_matrix)
    distributions = np.empty((number_of_steps + 1, transition_matrix.shape[0]), dtype=float)
    distributions[0] = initial_distribution
    for step in range(number_of_steps):
        distributions[step + 1] = distributions[step] @ transition_matrix
    return distributions


def calculate_stationary_distribution_for_finite_markov_chain(
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate stationary distribution pi satisfying pi P = pi."""

    validate_transition_matrix(transition_matrix)
    number_of_states = transition_matrix.shape[0]
    linear_system_matrix = transition_matrix.T - np.eye(number_of_states)
    linear_system_matrix[-1] = np.ones(number_of_states)
    right_hand_side = np.zeros(number_of_states)
    right_hand_side[-1] = 1.0
    stationary_distribution = np.linalg.solve(linear_system_matrix, right_hand_side)
    return stationary_distribution


def calculate_total_variation_distance_between_distributions(
    first_distribution: np.ndarray,
    second_distribution: np.ndarray,
) -> float:
    """Calculate total variation distance between two distributions."""

    return float(0.5 * np.abs(first_distribution - second_distribution).sum())


def calculate_total_variation_distances_to_stationarity(
    distributions: np.ndarray,
    stationary_distribution: np.ndarray,
) -> np.ndarray:
    """Calculate total variation distance to stationarity for every step."""

    return np.array(
        [
            calculate_total_variation_distance_between_distributions(distribution, stationary_distribution)
            for distribution in distributions
        ]
    )


def calculate_n_step_transition_matrix(
    transition_matrix: np.ndarray,
    number_of_steps: int,
) -> np.ndarray:
    """Calculate P raised to a given number of steps."""

    validate_transition_matrix(transition_matrix)
    return np.linalg.matrix_power(transition_matrix, number_of_steps)


def calculate_absorbing_chain_absorption_probabilities_and_expected_times(
    transition_matrix: np.ndarray,
    transient_state_indices: list[int],
    absorbing_state_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fundamental matrix, absorption probabilities, and expected absorption times."""

    validate_transition_matrix(transition_matrix)
    q_matrix = transition_matrix[np.ix_(transient_state_indices, transient_state_indices)]
    r_matrix = transition_matrix[np.ix_(transient_state_indices, absorbing_state_indices)]
    fundamental_matrix = np.linalg.inv(np.eye(len(transient_state_indices)) - q_matrix)
    absorption_probabilities = fundamental_matrix @ r_matrix
    expected_absorption_times = fundamental_matrix @ np.ones(len(transient_state_indices))
    return fundamental_matrix, absorption_probabilities, expected_absorption_times


def simulate_markov_modulated_returns(
    transition_matrix: np.ndarray,
    state_means: np.ndarray,
    state_volatilities: np.ndarray,
    initial_state: int,
    number_of_steps: int,
    number_of_paths: int,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate returns whose distribution depends on a Markov state."""

    rng = create_random_number_generator(random_seed)
    state_paths = simulate_multiple_markov_chain_paths(
        transition_matrix=transition_matrix,
        initial_state=initial_state,
        number_of_steps=number_of_steps,
        number_of_paths=number_of_paths,
        random_seed=random_seed,
    )
    returns = rng.normal(
        loc=state_means[state_paths[:, 1:]],
        scale=state_volatilities[state_paths[:, 1:]],
    )
    cumulative_returns = np.cumsum(returns, axis=1)
    return state_paths, returns, cumulative_returns


def plot_markov_chain_transition_matrix_heatmap(
    transition_matrix: np.ndarray,
    state_names: list[str],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot transition matrix as a heatmap."""

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(transition_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(state_names)))
    axis.set_yticks(range(len(state_names)))
    axis.set_xticklabels(state_names)
    axis.set_yticklabels(state_names)
    for row in range(transition_matrix.shape[0]):
        for column in range(transition_matrix.shape[1]):
            axis.text(column, row, f"{transition_matrix[row, column]:.2f}", ha="center", va="center")
    axis.set_title(title)
    axis.set_xlabel("Следующее состояние")
    axis.set_ylabel("Текущее состояние")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure, axis


def plot_markov_chain_sample_path(
    simulated_path: np.ndarray,
    state_names: list[str],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot one Markov chain state path."""

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.step(np.arange(simulated_path.size), simulated_path, where="post", linewidth=1.6)
    axis.set_yticks(range(len(state_names)))
    axis.set_yticklabels(state_names)
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Состояние")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_state_frequencies_and_theoretical_distributions(
    empirical_frequencies: np.ndarray,
    theoretical_distributions: np.ndarray,
    state_names: list[str],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot empirical and theoretical state probabilities over time."""

    figure, axis = plt.subplots(figsize=(11, 5))
    steps = np.arange(empirical_frequencies.shape[0])
    for state_index, state_name in enumerate(state_names):
        axis.plot(steps, empirical_frequencies[:, state_index], linewidth=2.0, label=f"{state_name}: сим.")
        axis.plot(
            steps,
            theoretical_distributions[:, state_index],
            "--",
            linewidth=1.5,
            label=f"{state_name}: теория",
        )
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Вероятность состояния")
    axis.legend(ncol=2)
    axis.grid(alpha=0.3)
    return figure, axis


def plot_total_variation_distance_to_stationarity(
    total_variation_distances: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot convergence to stationarity in total variation distance."""

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(np.arange(total_variation_distances.size), total_variation_distances, linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Расстояние полной вариации")
    axis.grid(alpha=0.3)
    return figure, axis


def plot_manual_markov_chain_graph(
    transition_matrix: np.ndarray,
    state_names: list[str],
    title: str,
) -> tuple[Figure, Axes]:
    """Plot a small Markov chain graph without external graph libraries."""

    number_of_states = len(state_names)
    angles = np.linspace(0.0, 2.0 * np.pi, number_of_states, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(positions[:, 0], positions[:, 1], s=1600, color="#f2f6ff", edgecolor="#4c78a8", linewidth=2)
    for state_index, state_name in enumerate(state_names):
        axis.text(positions[state_index, 0], positions[state_index, 1], state_name, ha="center", va="center")
    for row in range(number_of_states):
        for column in range(number_of_states):
            probability = transition_matrix[row, column]
            if probability <= 0.05:
                continue
            start = positions[row]
            end = positions[column]
            if row == column:
                axis.text(start[0] * 1.18, start[1] * 1.18, f"{probability:.2f}", ha="center", va="center")
            else:
                direction = end - start
                axis.arrow(
                    start[0] + 0.16 * direction[0],
                    start[1] + 0.16 * direction[1],
                    0.62 * direction[0],
                    0.62 * direction[1],
                    width=0.006,
                    head_width=0.06,
                    length_includes_head=True,
                    color="#4c78a8",
                    alpha=0.65,
                )
                midpoint = (start + end) / 2
                axis.text(midpoint[0], midpoint[1], f"{probability:.2f}", fontsize=9, ha="center", va="center")
    axis.set_title(title)
    axis.set_axis_off()
    axis.set_aspect("equal")
    return figure, axis


def plot_markov_modulated_cumulative_returns(
    cumulative_returns: np.ndarray,
    number_of_paths_to_plot: int,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot cumulative returns generated by Markov-modulated regimes."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for path_index in range(min(number_of_paths_to_plot, cumulative_returns.shape[0])):
        axis.plot(cumulative_returns[path_index], linewidth=1.2, alpha=0.85)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Шаг")
    axis.set_ylabel("Накопленная доходность")
    axis.grid(alpha=0.3)
    return figure, axis
