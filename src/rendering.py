"""
rendering.py

Renders an individual (array of 100 semi-transparent triangles) onto a blank
canvas and returns the resulting RGB image as a NumPy array.

Pipeline:
    1. Start with a black canvas (300 x 400 x 3, uint8).
    2. Iterate through triangles front-to-back order index 0 → 99.
    3. For each triangle, draw it onto the canvas using alpha compositing.

Uses OpenCV (cv2.fillPoly) for fast rasterisation.

Author: Miguel (Member B — Rendering & Fitness)
"""

import cv2
import numpy as np

from src.representation import IMAGE_WIDTH, IMAGE_HEIGHT, NUM_TRIANGLES

DOWNSCALE_WIDTH = 150
DOWNSCALE_HEIGHT = 200


def render_individual(individual: np.ndarray) -> np.ndarray:
    """
    Render an individual to a (H, W, 3) uint8 RGB image.

    Parameters
    ----------
    individual : np.ndarray, shape (num_triangles, 10)
        Each row: [x1, y1, x2, y2, x3, y3, r, g, b, alpha]

    Returns
    -------
    canvas : np.ndarray, shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype uint8
        The rendered image in RGB colour space.
    """
    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float64)

    for tri in individual:
        x1, y1, x2, y2, x3, y3 = tri[0:6]
        r, g, b, alpha = tri[6], tri[7], tri[8], tri[9]

        # Build triangle vertices as integer pixel coordinates (x, y)
        pts = np.array([
            [int(round(x1)), int(round(y1))],
            [int(round(x2)), int(round(y2))],
            [int(round(x3)), int(round(y3))],
        ], dtype=np.int32)

        # Create a single-triangle mask
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        # Alpha-composite: canvas = canvas * (1 - alpha) + color * alpha
        color = np.array([r, g, b], dtype=np.float64)
        region = mask.astype(bool)
        canvas[region] = canvas[region] * (1.0 - alpha) + color * alpha

    return np.clip(canvas, 0, 255).astype(np.uint8)


def render_individual_fast(individual: np.ndarray) -> np.ndarray:
    """
    Faster rendering using a pre-allocated overlay buffer.

    Same interface as render_individual but avoids repeated boolean indexing
    by using cv2's addWeighted on full-size overlays. Faster for dense
    triangles that cover large portions of the canvas.

    Parameters
    ----------
    individual : np.ndarray, shape (num_triangles, 10)

    Returns
    -------
    canvas : np.ndarray, shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype uint8
    """
    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    for tri in individual:
        x1, y1, x2, y2, x3, y3 = tri[0:6]
        r, g, b, alpha = float(tri[6]), float(tri[7]), float(tri[8]), float(tri[9])

        pts = np.array([
            [int(round(x1)), int(round(y1))],
            [int(round(x2)), int(round(y2))],
            [int(round(x3)), int(round(y3))],
        ], dtype=np.int32)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (int(round(r)), int(round(g)), int(round(b))))
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    return canvas


def render_individual_downscaled(individual: np.ndarray) -> np.ndarray:
    """
    Render at half resolution (150x200) for faster fitness evaluation.
    Vertex coordinates are scaled by 0.5.
    """
    canvas = np.zeros((DOWNSCALE_HEIGHT, DOWNSCALE_WIDTH, 3), dtype=np.uint8)

    for tri in individual:
        x1, y1, x2, y2, x3, y3 = tri[0:6] * 0.5
        r, g, b, alpha = float(tri[6]), float(tri[7]), float(tri[8]), float(tri[9])

        pts = np.array([
            [int(round(x1)), int(round(y1))],
            [int(round(x2)), int(round(y2))],
            [int(round(x3)), int(round(y3))],
        ], dtype=np.int32)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (int(round(r)), int(round(g)), int(round(b))))
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    return canvas
