"""Problem definitions for active constraint-learning experiments.

The sign convention matches the paper: g(kappa) <= 0 is safe, and
g(kappa) > 0 is inside the avoid set.
"""

from __future__ import annotations

import numpy as np


BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=float)


def g_circle(points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(points).astype(float)
    center = np.array([0.0, 0.0])
    radius = 0.62
    return radius**2 - np.sum((pts - center) ** 2, axis=1)


def g_wavy(points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(points).astype(float)
    x = pts[:, 0]
    y = pts[:, 1]
    theta = np.arctan2(y, x)
    radius = wavy_radius(theta)
    return radius**2 - (x**2 + y**2)


def wavy_radius(theta: np.ndarray) -> np.ndarray:
    return 1.05 + 0.20 * np.sin(3.0 * theta) + 0.13 * np.cos(5.0 * theta)


CONSTRAINTS = {
    "circle": {
        "name": "circle",
        "function": g_circle,
        "bounds": BOUNDS,
        "paper_reference": "scaled 2D analogue of Appendix B obstacle recovery",
    },
    "wavy": {
        "name": "wavy",
        "function": g_wavy,
        "bounds": BOUNDS,
        "paper_reference": "nonlinear 2D obstacle boundary",
    },
}


def labels_for(points: np.ndarray, constraint_name: str = "circle") -> np.ndarray:
    return (CONSTRAINTS[constraint_name]["function"](points) <= 0.0).astype(int)


def make_query_grid(n_per_axis: int = 75) -> np.ndarray:
    """Unlabeled candidate states for the active query optimizer.

    This grid is deliberately broad and does not use the true constraint
    boundary; it is part of the method's search space, not the score.
    """
    radii = np.linspace(0.55, 1.60, n_per_axis)
    angles = np.linspace(-np.pi, np.pi, n_per_axis, endpoint=False)
    rr, tt = np.meshgrid(radii, angles)
    return np.column_stack([rr.ravel() * np.cos(tt.ravel()), rr.ravel() * np.sin(tt.ravel())])


def make_eval_grid(n_per_axis: int = 61, constraint_name: str = "wavy") -> np.ndarray:
    """Boundary-focused evaluation points for obstacle recovery.

    The paper's reported errors concentrate near the unsafe-set boundary. A
    full-square grid made the scaled synthetic task trivial, so the score uses
    a fixed test band around the known benchmark boundary. These coordinates
    are only used for final prediction artifacts, never to choose active
    queries.
    """
    angles = np.linspace(-np.pi, np.pi, n_per_axis, endpoint=False)
    offsets = np.linspace(-0.12, 0.12, n_per_axis)
    aa, oo = np.meshgrid(angles, offsets)
    if constraint_name == "circle":
        radius = 0.62 + oo
    else:
        radius = wavy_radius(aa) + oo
    radius = np.maximum(radius, 0.05)
    return np.column_stack([radius.ravel() * np.cos(aa.ravel()), radius.ravel() * np.sin(aa.ravel())])


def default_initial_pairs() -> list[tuple[list[float], list[float]]]:
    return [
        ([-1.55, -1.10], [1.55, 1.10]),
        ([-1.55, 1.10], [1.55, -1.10]),
        ([-1.75, 0.25], [1.75, 0.25]),
        ([0.25, -1.75], [0.25, 1.75]),
    ]
