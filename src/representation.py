"""
representation.py

Defines the genetic representation for the triangle-based image reconstruction problem.

Representation:
    - Individual: np.ndarray of shape (100, 10)
    - Triangle (row): [x1, y1, x2, y2, x3, y3, r, g, b, alpha]

Gene ranges:
    - image size = 300x400, x ∈ [0, 299], y ∈ [0, 399]
    - RGB ∈ [0, 255]
    - alpha ∈ [0.0, 1.0]

Responsibilities:
    - Generate random triangles and individuals
    - Enforce valid gene ranges (clamping)
    - Provide consistent representation across GA components

Used by:
    - rendering.py (to draw triangles)
    - crossover.py / mutation.py (to modify individuals)
    - ga.py (population initialization)
"""
import numpy as np

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 400
NUM_TRIANGLES = 100
GENES_PER_TRIANGLE = 10

X_MIN, X_MAX = 0.0, float(IMAGE_WIDTH - 1)   
Y_MIN, Y_MAX = 0.0, float(IMAGE_HEIGHT - 1) 
RGB_MIN, RGB_MAX = 0.0, 255.0
ALPHA_MIN, ALPHA_MAX = 0.1, 0.8

"""
    Generate a single triangle encoded as a vector of 10 genes.
"""
def random_triangle(rng: np.random.Generator) -> np.ndarray:
    triangle = np.empty(GENES_PER_TRIANGLE, dtype=np.float32)

    triangle[0] = rng.uniform(X_MIN, X_MAX) 
    triangle[1] = rng.uniform(Y_MIN, Y_MAX) 
    triangle[2] = rng.uniform(X_MIN, X_MAX) 
    triangle[3] = rng.uniform(Y_MIN, Y_MAX) 
    triangle[4] = rng.uniform(X_MIN, X_MAX)  
    triangle[5] = rng.uniform(Y_MIN, Y_MAX) 

    triangle[6] = rng.uniform(RGB_MIN, RGB_MAX) 
    triangle[7] = rng.uniform(RGB_MIN, RGB_MAX) 
    triangle[8] = rng.uniform(RGB_MIN, RGB_MAX) 
    triangle[9] = rng.uniform(ALPHA_MIN, ALPHA_MAX) 

    return triangle


"""
Generate a random individual (set of triangles).
Returns: np.ndarray of shape (num_triangles, 10)
"""
def random_individual(
    rng: np.random.Generator,
    num_triangles: int = NUM_TRIANGLES
) -> np.ndarray:
    return np.array(
        [random_triangle(rng) for _ in range(num_triangles)],
        dtype=np.float32
    )


"""
    Clamp all gene values to valid ranges.
"""
def clamp_individual(individual: np.ndarray) -> np.ndarray:
    clamped = individual.copy()

    clamped[:, 0] = np.clip(clamped[:, 0], X_MIN, X_MAX)
    clamped[:, 1] = np.clip(clamped[:, 1], Y_MIN, Y_MAX)
    clamped[:, 2] = np.clip(clamped[:, 2], X_MIN, X_MAX)
    clamped[:, 3] = np.clip(clamped[:, 3], Y_MIN, Y_MAX)
    clamped[:, 4] = np.clip(clamped[:, 4], X_MIN, X_MAX)
    clamped[:, 5] = np.clip(clamped[:, 5], Y_MIN, Y_MAX)

    clamped[:, 6] = np.clip(clamped[:, 6], RGB_MIN, RGB_MAX)
    clamped[:, 7] = np.clip(clamped[:, 7], RGB_MIN, RGB_MAX)
    clamped[:, 8] = np.clip(clamped[:, 8], RGB_MIN, RGB_MAX)

    clamped[:, 9] = np.clip(clamped[:, 9], ALPHA_MIN, ALPHA_MAX)

    return clamped
