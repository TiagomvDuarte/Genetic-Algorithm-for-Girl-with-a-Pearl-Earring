"""
Selection operators for the triangle-based genetic algorithm are responsible for choosing individuals from the population to act as parents for the next generation, based on their fitness.
Implemented methods in our project:
    - Tournament selection
    - Roulette wheel selection
    - Rank-based selection

This project minimizes fitness (lower RMSE is better) - lower-fitness individuals have higher selection probability
"""

import numpy as np


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    k: int = 3
) -> np.ndarray:
    """
    Tournament selection - pick a few random individuals, choose the best among them
Parameters:
    population: np.ndarray of individuals, shape (pop_size, ...)
    fitness: np.ndarray of fitness values, shape (pop_size,)
    rng: NumPy random generator
    k: tournament size
Returns a copy of the selected individual
    """
    pop_size = len(population)

    if pop_size == 0:
        raise ValueError("Population must not be empty.")
    if len(fitness) != pop_size:
        raise ValueError("Population and fitness must have the same length.")
    if k <= 0:
        raise ValueError("Tournament size k must be > 0.")

    k = min(k, pop_size)
    indices = rng.choice(pop_size, size=k, replace=False)
    best_idx = indices[np.argmin(fitness[indices])]

    return population[best_idx].copy()


def roulette_wheel_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Roulette selection - probability based on fitness, better = higher chance
    Since this is a minimization problem, lower fitness should mean higher probability. We convert fitness into weights using inverse fitness.

    Parameters:
        population: np.ndarray of individuals, shape (pop_size, ...)
        fitness: np.ndarray of fitness values, shape (pop_size,)
        rng: NumPy random generator
        epsilon: small value to avoid division by zero

    Returns a copy of the selected individual
    """
    pop_size = len(population)

    if pop_size == 0:
        raise ValueError("Population must not be empty.")
    if len(fitness) != pop_size:
        raise ValueError("Population and fitness must have the same length.")

    fitness = np.asarray(fitness, dtype=np.float64)

    if np.any(fitness < 0): # RMSE should not be negative
        fitness = fitness - fitness.min()

    weights = 1.0 / (fitness + epsilon)
    probabilities = weights / weights.sum()

    selected_idx = rng.choice(pop_size, p=probabilities)
    return population[selected_idx].copy()


def rank_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Rank-based selection - sort population, assign probability based on rank
    The best individual gets the highest rank weight, and probabilities are assigned based on rank rather than raw fitness.

    Parameters:
        population: np.ndarray of individuals, shape (pop_size, ...)
        fitness: np.ndarray of fitness values, shape (pop_size,)
        rng: NumPy random generator

    Returns a copy of the selected individual
    """
    pop_size = len(population)

    if pop_size == 0:
        raise ValueError("Population must not be empty.")
    if len(fitness) != pop_size:
        raise ValueError("Population and fitness must have the same length.")

    sorted_indices = np.argsort(fitness)  

    rank_weights = np.arange(pop_size, 0, -1, dtype=np.float64)
    probabilities = rank_weights / rank_weights.sum()

    selected_sorted_pos = rng.choice(pop_size, p=probabilities)
    selected_idx = sorted_indices[selected_sorted_pos]

    return population[selected_idx].copy()


def select_parent(
    population: np.ndarray,
    fitness: np.ndarray,
    method: str = "tournament",
    rng: np.random.Generator | None = None,
    k: int = 3
) -> np.ndarray:
    """
    Unified wrapper for parent selection.

    Parameters:
        population: np.ndarray of individuals, shape (pop_size, ...)
        fitness: np.ndarray of fitness values, shape (pop_size,)
        method: selection method ("tournament", "roulette", "rank")
        rng: NumPy random generator; if None, a new generator is created
        k: tournament size for tournament selection

    Returns a copy of the selected individual
    """
    if rng is None:
        rng = np.random.default_rng()

    method = method.lower()

    if method == "tournament":
        return tournament_selection(population, fitness, rng, k=k)
    if method == "roulette":
        return roulette_wheel_selection(population, fitness, rng)
    if method == "rank":
        return rank_selection(population, fitness, rng)
    if method == "random":
        return population[rng.integers(0, len(population))].copy()

    raise ValueError(
        f"Unknown selection method: {method}. "
        f"Choose from 'tournament', 'roulette', 'rank', or 'random'."
    )


def select_parent_with_index(
    population: np.ndarray,
    fitness: np.ndarray,
    method: str = "tournament",
    rng: np.random.Generator | None = None,
    k: int = 3
) -> tuple:
    """
    Select a parent and return (individual_copy, index) so the caller
    can look up the parent's fitness value.
    """
    if rng is None:
        rng = np.random.default_rng()

    method = method.lower()

    if method == "tournament":
        pop_size = len(population)
        k = min(k, pop_size)
        indices = rng.choice(pop_size, size=k, replace=False)
        best_idx = indices[np.argmin(fitness[indices])]
        return population[best_idx].copy(), int(best_idx)

    if method == "roulette":
        pop_size = len(population)
        fit = np.asarray(fitness, dtype=np.float64)
        if np.any(fit < 0):
            fit = fit - fit.min()
        weights = 1.0 / (fit + 1e-8)
        probabilities = weights / weights.sum()
        idx = rng.choice(pop_size, p=probabilities)
        return population[idx].copy(), int(idx)

    if method == "rank":
        pop_size = len(population)
        sorted_indices = np.argsort(fitness)
        rank_weights = np.arange(pop_size, 0, -1, dtype=np.float64)
        probabilities = rank_weights / rank_weights.sum()
        selected_pos = rng.choice(pop_size, p=probabilities)
        idx = sorted_indices[selected_pos]
        return population[idx].copy(), int(idx)

    if method == "random":
        idx = int(rng.integers(0, len(population)))
        return population[idx].copy(), idx

    raise ValueError(f"Unknown selection method: {method}.")