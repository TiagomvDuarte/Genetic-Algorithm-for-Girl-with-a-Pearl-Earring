"""
crossover.py

Crossover operators for the triangle-based genetic algorithm.

Triangle-level (Anastasiia):
    - Triangle-level single-point crossover
    - Uniform triangle swap crossover

Gene-level:
    - Gene-level single-point crossover
    - Gene-level uniform crossover
"""

import numpy as np


def triangle_single_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Split at triangle index i, child gets triangles 0..i from parent A and i+1..99 from parent B
    """
    num_triangles = parent_a.shape[0]

    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    i = rng.integers(1, num_triangles)

    child = np.empty_like(parent_a)

    child[:i] = parent_a[:i]
    child[i:] = parent_b[i:]

    return child


def uniform_triangle_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Uniform triangle crossover.

    Each triangle is independently inherited from parent A or B
    with probability 0.5.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    num_triangles = parent_a.shape[0]
    mask = rng.random(num_triangles) < 0.5

    child = np.where(mask[:, None], parent_a, parent_b)

    return child


def gene_single_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Gene-level single-point crossover.

    Flattens both parents into a 1D gene vector (1000 genes), picks a random
    cut point, and takes genes 0..cut from parent A and cut+1..end from parent B.
    This allows mixing *within* triangles, not just swapping whole triangles.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    flat_a = parent_a.flatten()
    flat_b = parent_b.flatten()
    total_genes = len(flat_a)

    cut = rng.integers(1, total_genes)

    child_flat = np.empty_like(flat_a)
    child_flat[:cut] = flat_a[:cut]
    child_flat[cut:] = flat_b[cut:]

    return child_flat.reshape(parent_a.shape)


def gene_uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Gene-level uniform crossover.

    Each gene is independently inherited from parent A or B with probability 0.5.
    Operates on the flattened gene vector so mixing happens within triangles.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    mask = rng.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b)

    return child


def blend_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Blend crossover: each gene is interpolated between parents using a
    random alpha drawn from [0.3, 0.7] per gene. Creates genuinely new
    gene values rather than just recombining existing ones.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    alphas = rng.uniform(0.3, 0.7, parent_a.shape).astype(np.float32)
    child = alphas * parent_a + (1.0 - alphas) * parent_b
    return child


def fitness_weighted_blend_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    fitness_a: float = None,
    fitness_b: float = None,
) -> np.ndarray:
    """
    Blend crossover where the fitter (lower fitness) parent contributes more.
    If fitness values aren't provided, falls back to equal blend.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError("Parents must have the same shape.")

    if fitness_a is None or fitness_b is None:
        return blend_crossover(parent_a, parent_b, rng)

    total = fitness_a + fitness_b
    if total < 1e-8:
        alpha = 0.5
    else:
        alpha = fitness_b / total  # lower fitness (better) parent gets higher weight

    child = alpha * parent_a + (1.0 - alpha) * parent_b
    return child.astype(np.float32)


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    method: str = "single_point",
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """
    Wrapper for crossover methods.
    """
    if rng is None:
        rng = np.random.default_rng()

    method = method.lower()

    if method == "single_point":
        return triangle_single_point_crossover(parent_a, parent_b, rng)

    if method == "uniform":
        return uniform_triangle_crossover(parent_a, parent_b, rng)

    if method == "gene_single_point":
        return gene_single_point_crossover(parent_a, parent_b, rng)

    if method == "gene_uniform":
        return gene_uniform_crossover(parent_a, parent_b, rng)

    if method == "blend":
        return blend_crossover(parent_a, parent_b, rng)

    if method == "fitness_weighted_blend":
        return fitness_weighted_blend_crossover(
            parent_a, parent_b, rng,
            fitness_a=kwargs.get("fitness_a"),
            fitness_b=kwargs.get("fitness_b"),
        )

    raise ValueError(
        "Unknown crossover method. Use 'single_point', 'uniform', "
        "'gene_single_point', 'gene_uniform', 'blend', or 'fitness_weighted_blend'."
    )