"""
fitness.py

Fitness evaluation for the triangle-based image reconstruction GA.

Primary metric — RMSE (Root Mean Squared Error):
    RMSE = sqrt( (1 / (W * H * 3)) * sum((pixel_original - pixel_generated)^2) )

Computed pixel-by-pixel across all 3 RGB channels over the 300x400 image.
Lower is better.

Author: Miguel (Member B — Rendering & Fitness)
"""

import cv2
import numpy as np
from PIL import Image

from src.representation import IMAGE_WIDTH, IMAGE_HEIGHT
from src.rendering import (
    DOWNSCALE_HEIGHT,
    DOWNSCALE_WIDTH,
    render_individual_fast as render_individual,
    render_individual_downscaled,
)


# ---------------------------------------------------------------------------
# Target image loading
# ---------------------------------------------------------------------------

_target_cache: dict = {}


def load_target(path: str = "data/target.png") -> np.ndarray:
    """
    Load and cache the target image as a (H, W, 3) uint8 RGB array.
    """
    if path not in _target_cache:
        img = Image.open(path).convert("RGB")
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        _target_cache[path] = np.array(img, dtype=np.uint8)
    return _target_cache[path]


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Compute RMSE between two (H, W, 3) uint8 images.

    Parameters
    ----------
    rendered : np.ndarray, shape (H, W, 3)
    target   : np.ndarray, shape (H, W, 3)

    Returns
    -------
    float
        RMSE value (lower is better). Range roughly [0, 255].
    """
    diff = rendered.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Fitness function (used by the GA)
# ---------------------------------------------------------------------------

def evaluate(individual: np.ndarray, target: np.ndarray | None = None, fitness_mode: str = "rmse") -> float:
    """
    Evaluate the fitness of an individual.

    Renders the individual to an image and computes RMSE against the target.

    Parameters
    ----------
    individual : np.ndarray, shape (num_triangles, 10)
    target     : np.ndarray, shape (H, W, 3), optional
        If None, the default target image is loaded from disk.

    Returns
    -------
    float
        Fitness value (RMSE — lower is better).
    """
    if target is None:
        target = load_target()
    rendered = render_individual(individual)
    if fitness_mode == "hybrid":
        return hybrid_fitness(rendered, target)
    if fitness_mode == "perceptual":
        return perceptual_rmse(rendered, target)
    return rmse(rendered, target)


# ---------------------------------------------------------------------------
# Alternative fitness functions
# ---------------------------------------------------------------------------

def hybrid_fitness(rendered: np.ndarray, target: np.ndarray,
                   rmse_weight: float = 0.7, ssim_weight: float = 0.3) -> float:
    """
    Weighted combination of RMSE and SSIM loss.
    Formula: rmse_weight * RMSE + ssim_weight * (1 - SSIM) * 255
    """
    from skimage.metrics import structural_similarity as ssim_metric
    rmse_val = rmse(rendered, target)
    ssim_val = ssim_metric(rendered, target, channel_axis=2, data_range=255)
    ssim_loss = (1.0 - ssim_val) * 255.0
    return rmse_weight * rmse_val + ssim_weight * ssim_loss


def perceptual_rmse(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    RMSE with perceptual weighting based on ITU-R BT.601 luminance coefficients.
    Green (0.587) > Red (0.299) > Blue (0.114).
    """
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    diff = rendered.astype(np.float64) - target.astype(np.float64)
    weighted_diff = diff * weights[np.newaxis, np.newaxis, :]
    return float(np.sqrt(np.mean(weighted_diff ** 2)))


# ---------------------------------------------------------------------------
# Downscaled (fast) evaluation
# ---------------------------------------------------------------------------

_target_downscaled_cache: dict = {}


def downscale_target(target: np.ndarray) -> np.ndarray:
    """Return the half-resolution version of an already loaded target image."""
    return cv2.resize(target, (DOWNSCALE_WIDTH, DOWNSCALE_HEIGHT), interpolation=cv2.INTER_AREA)


def load_target_downscaled(path: str = "data/target.png") -> np.ndarray:
    """Load and cache a downscaled (150x200) version of the target image."""
    if path not in _target_downscaled_cache:
        full = load_target(path)
        _target_downscaled_cache[path] = downscale_target(full)
    return _target_downscaled_cache[path]


def evaluate_fast(individual: np.ndarray, target_downscaled: np.ndarray | None = None,
                  fitness_mode: str = "rmse") -> float:
    """
    Evaluate fitness using half-resolution rendering (4x fewer pixels).
    """
    if target_downscaled is None:
        target_downscaled = load_target_downscaled()
    rendered = render_individual_downscaled(individual)
    if fitness_mode == "hybrid":
        return hybrid_fitness(rendered, target_downscaled)
    if fitness_mode == "perceptual":
        return perceptual_rmse(rendered, target_downscaled)
    return rmse(rendered, target_downscaled)
