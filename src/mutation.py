"""
mutation.py

Mutation operators for the triangle-based image reconstruction GA.

Operators:
    - gaussian_mutation   : perturbs all genes with Gaussian noise (vertex + color)
    - vertex_jitter       : perturbs only vertex coordinates (columns 0-5)
    - color_perturbation  : perturbs only RGB/alpha channels (columns 6-9)
    - triangle_reset      : replaces entire triangles with new random ones
    - swap_mutation       : swaps rendering order of two random triangles
    - adaptive_mutation   : gaussian mutation with sigma that decays over generations

All operators return a new individual (do not modify in-place).
All operators call clamp_individual() to enforce valid gene ranges.

Author: Tiago (Member B — Mutation & Experiments)
"""

import numpy as np

from src.representation import (
    clamp_individual,
    random_triangle,
)


def _alpha_sigma(sigma: float) -> float:
    """Scale RGB-sized mutation sigma to the alpha gene's 0-1 range."""
    return sigma / 255.0


def gaussian_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 10.0,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Perturb individual genes with Gaussian noise.
    Each gene is mutated independently with probability mutation_rate.
    A random Gaussian offset (mean=0, std=sigma) is added to the gene.

    Parameters
    ----------
    individual    : np.ndarray, shape (num_triangles, 10)
    rng           : np.random.Generator
    sigma         : standard deviation of the Gaussian noise
    mutation_rate : probability of mutating each gene

    Returns
    -------
    np.ndarray, shape (num_triangles, 10)
    """
    mutant = individual.copy()

    genes = mutant[:, 0:9]
    gene_mask = rng.random(genes.shape) < mutation_rate
    gene_noise = rng.normal(0.0, sigma, genes.shape)
    genes[gene_mask] += gene_noise[gene_mask]
    mutant[:, 0:9] = genes

    alpha = mutant[:, 9]
    alpha_mask = rng.random(alpha.shape) < mutation_rate
    alpha_noise = rng.normal(0.0, _alpha_sigma(sigma), alpha.shape)
    alpha[alpha_mask] += alpha_noise[alpha_mask]
    mutant[:, 9] = alpha

    return clamp_individual(mutant)


def vertex_jitter(
    individual: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 10.0,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Perturb only vertex coordinates (columns 0-5) with Gaussian noise.
    Color and alpha are left unchanged.

    Parameters
    ----------
    individual    : np.ndarray, shape (num_triangles, 10)
    rng           : np.random.Generator
    sigma         : standard deviation of the Gaussian noise
    mutation_rate : probability of mutating each coordinate gene

    Returns
    -------
    np.ndarray, shape (num_triangles, 10)
    """
    mutant = individual.copy()
    coords = mutant[:, 0:6]
    mask = rng.random(coords.shape) < mutation_rate
    noise = rng.normal(0.0, sigma, coords.shape)
    coords[mask] += noise[mask]
    mutant[:, 0:6] = coords
    return clamp_individual(mutant)


def color_perturbation(
    individual: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 10.0,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Perturb only RGB and alpha channels (columns 6-9) with Gaussian noise.
    Vertex coordinates are left unchanged.

    Parameters
    ----------
    individual    : np.ndarray, shape (num_triangles, 10)
    rng           : np.random.Generator
    sigma         : standard deviation of the Gaussian noise
    mutation_rate : probability of mutating each color gene

    Returns
    -------
    np.ndarray, shape (num_triangles, 10)
    """
    mutant = individual.copy()

    rgb = mutant[:, 6:9]
    rgb_mask = rng.random(rgb.shape) < mutation_rate
    rgb_noise = rng.normal(0.0, sigma, rgb.shape)
    rgb[rgb_mask] += rgb_noise[rgb_mask]
    mutant[:, 6:9] = rgb

    alpha = mutant[:, 9]
    alpha_mask = rng.random(alpha.shape) < mutation_rate
    alpha_noise = rng.normal(0.0, _alpha_sigma(sigma), alpha.shape)
    alpha[alpha_mask] += alpha_noise[alpha_mask]
    mutant[:, 9] = alpha

    return clamp_individual(mutant)


def triangle_reset(
    individual: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.05,
) -> np.ndarray:
    """
    Replace entire triangles with new random ones.
    Each triangle is replaced with probability mutation_rate.
    Useful for escaping local optima when a triangle is stuck
    in an irrelevant region of the image.

    Parameters
    ----------
    individual    : np.ndarray, shape (num_triangles, 10)
    rng           : np.random.Generator
    mutation_rate : probability of replacing each triangle

    Returns
    -------
    np.ndarray, shape (num_triangles, 10)
    """
    mutant = individual.copy()
    for i in range(len(mutant)):
        if rng.random() < mutation_rate:
            mutant[i] = random_triangle(rng)
    return mutant


def swap_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.05,
) -> np.ndarray:
    """
    Swap the rendering order of two random triangles.
    Since triangles are drawn front-to-back (index 0 first),
    order affects which triangles occlude others.

    Each pair of triangles is swapped with probability mutation_rate.

    Parameters
    ----------
    individual    : np.ndarray, shape (num_triangles, 10)
    rng           : np.random.Generator
    mutation_rate : probability of performing each swap

    Returns
    -------
    np.ndarray, shape (num_triangles, 10)
    """
    mutant = individual.copy()
    n = len(mutant)
    for i in range(n):
        if rng.random() < mutation_rate:
            j = rng.integers(0, n)
            mutant[[i, j]] = mutant[[j, i]]
    return mutant


def adaptive_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    generation: int,
    max_generations: int,
    sigma_initial: float = 20.0,
    sigma_final: float = 1.0,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Gaussian mutation with sigma that decays linearly over generations.
    Early generations: large sigma -> broad exploration
    Late generations:  small sigma -> fine-grained refinement

    Parameters
    ----------
    individual      : np.ndarray, shape (num_triangles, 10)
    rng             : np.random.Generator
    generation      : current generation (0-indexed)
    max_generations : total number of generations
    sigma_initial   : starting sigma (large, for exploration)
    sigma_final     : ending sigma (small, for refinement)
    mutation_rate   : probability of mutating each gene

    Returns
    -------
    np.ndarray, shape (num_triangles, 10)
    """
    sigma = current_sigma(generation, max_generations, sigma_initial, sigma_final)
    return gaussian_mutation(individual, rng, sigma=sigma, mutation_rate=mutation_rate)


def multi_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 10.0,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Apply vertex_jitter + color_perturbation together.
    Each sub-operator is applied with 50% probability,
    guaranteeing at least one is always applied.
    """
    apply_vertex = rng.random() < 0.5
    apply_color = rng.random() < 0.5
    if not apply_vertex and not apply_color:
        apply_vertex = True

    mutant = individual.copy()
    if apply_vertex:
        mutant = vertex_jitter(mutant, rng=rng, sigma=sigma, mutation_rate=mutation_rate)
    if apply_color:
        mutant = color_perturbation(mutant, rng=rng, sigma=sigma, mutation_rate=mutation_rate)
    return mutant


def sort_by_area_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.05,
) -> np.ndarray:
    """
    Sort triangles by area (largest first, smallest last) with probability mutation_rate.
    Larger triangles rendered first (background), smaller triangles on top (detail).
    """
    if rng.random() > mutation_rate:
        return individual.copy()

    mutant = individual.copy()
    areas = 0.5 * np.abs(
        (mutant[:, 2] - mutant[:, 0]) * (mutant[:, 5] - mutant[:, 1]) -
        (mutant[:, 4] - mutant[:, 0]) * (mutant[:, 3] - mutant[:, 1])
    )
    sorted_indices = np.argsort(-areas)
    return mutant[sorted_indices]


def adaptive_vertex_jitter(
    individual: np.ndarray,
    rng: np.random.Generator,
    generation: int,
    max_generations: int,
    sigma_initial: float = 20.0,
    sigma_final: float = 1.0,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Vertex jitter with linearly decaying sigma for exploration-to-refinement transition.
    """
    sigma = current_sigma(generation, max_generations, sigma_initial, sigma_final)
    return vertex_jitter(individual, rng=rng, sigma=sigma, mutation_rate=mutation_rate)


def current_sigma(
    generation: int,
    max_generations: int,
    sigma_initial: float = 20.0,
    sigma_final: float = 1.0,
) -> float:
    """
    Returns the sigma value for a given generation.
    Useful for convergence logging.
    """
    t = generation / max_generations
    return sigma_initial + (sigma_final - sigma_initial) * t
