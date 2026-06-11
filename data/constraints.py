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
    radius = 0.58 + 0.10 * np.sin(3.0 * np.arctan2(y, x))
    return radius**2 - (x**2 + y**2)


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


def make_eval_grid(n_per_axis: int = 61) -> np.ndarray:
    xs = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], n_per_axis)
    ys = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], n_per_axis)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def default_initial_pairs() -> list[tuple[list[float], list[float]]]:
    return [
        ([-1.55, -1.10], [1.55, 1.10]),
        ([-1.55, 1.10], [1.55, -1.10]),
        ([-1.75, 0.25], [1.75, 0.25]),
        ([0.25, -1.75], [0.25, 1.75]),
    ]
