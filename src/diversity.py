"""
diversity.py

Diversity-preservation mechanisms for the triangle-based genetic algorithm.

    - Fitness Sharing: penalizes individuals in dense regions of the population,
      encouraging exploration of multiple optima.

    - Restricted Mating (Best Partial Match): when selecting a second parent,
      picks the most similar individual from a random pool, keeping niches intact.
"""

import numpy as np


def compute_pairwise_distances(population: np.ndarray) -> np.ndarray:
    """
    Compute normalized Euclidean distances between all pairs of individuals.

    Returns a (pop_size, pop_size) matrix with values in [0, 1].
    """
    pop_size = len(population)
    flat = population.reshape(pop_size, -1).astype(np.float64)

    # Euclidean distances via broadcasting
    diff = flat[:, np.newaxis, :] - flat[np.newaxis, :, :]      # (N, N, genes)
    raw_distances = np.sqrt((diff ** 2).sum(axis=-1))            # (N, N)

    max_dist = raw_distances.max()
    if max_dist > 0:
        return raw_distances / max_dist
    return raw_distances


def apply_fitness_sharing(
    fitness: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """
    Apply fitness sharing to penalize individuals in crowded regions.

    S(i,j) = 1 - d_normalized(i,j)
    SC(i)  = sum_j S(i,j)
    f_shared(i) = f(i) * SC(i)

    For minimization problems, a crowded individual gets its fitness multiplied
    by its sharing coefficient SC(i) >= 1, making it worse (higher).
    """
    distances = compute_pairwise_distances(population)   # (N, N), normalized

    # S(i,j) = 1 - d_normalized(i,j)
    similarity = 1.0 - distances                         # (N, N)

    # SC(i) = sum_j S(i,j)  — always >= 1 because S(i,i) = 1
    sharing_coefficients = similarity.sum(axis=1)        # (N,)

    shared_fitness = fitness * sharing_coefficients

    return shared_fitness.astype(np.float32)


def select_parent_restricted_mating(
    population: np.ndarray,
    fitness: np.ndarray,
    anchor: np.ndarray,
    rng: np.random.Generator,
    pool_size: int = 5,
) -> np.ndarray:
    """
    Restricted mating (Best Partial Match variant).

    Given an already-selected parent (anchor), sample a pool of candidates
    and return the one most similar to the anchor. This keeps niches intact
    by preferring to mate similar individuals.

    Never blocks reproduction — always returns someone.

    Parameters
    ----------
    population : np.ndarray
        Full population, shape (pop_size, ...).
    fitness : np.ndarray
        Fitness array (unused here, kept for API consistency).
    anchor : np.ndarray
        The first parent already selected.
    rng : np.random.Generator
        NumPy random generator.
    pool_size : int
        Number of candidates to sample before picking the most similar.

    Returns
    -------
    np.ndarray
        Copy of the selected second parent.
    """
    pop_size = len(population)
    pool_size = min(pool_size, pop_size)

    candidate_indices = rng.choice(pop_size, size=pool_size, replace=False)

    anchor_flat = anchor.flatten().astype(np.float64)
    best_idx = None
    best_similarity = -1.0

    for idx in candidate_indices:
        candidate_flat = population[idx].flatten().astype(np.float64)
        dist = np.linalg.norm(anchor_flat - candidate_flat)
        # Higher similarity = lower distance
        similarity = 1.0 / (1.0 + dist)
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = idx

    return population[best_idx].copy()
