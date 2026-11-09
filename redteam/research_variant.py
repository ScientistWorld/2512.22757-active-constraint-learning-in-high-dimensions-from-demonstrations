"""Honest audit variants. No test labels or evaluator implementation is read.

Reference acquisition is unchanged. The better variant fits a periodic boundary
to KKT contact positions/normals from those same trajectories. It never uses
the obstacle function for fitting. Access to that function stays in the supplied
demonstration simulator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.constraints import make_eval_grid
from method import gp_acl as acl


def fit_boundary(demos: list, output: Path, seed: int, simplify: bool = False) -> None:
    points, normals = [], []
    for path in demos:
        p, n = acl.kkt_tight_points(np.asarray(path))
        if len(p):
            points.extend(p)
            normals.extend(n)
    points, normals = np.asarray(points), np.asarray(normals)
    theta = np.arctan2(points[:, 1], points[:, 0])
    radii = np.linalg.norm(points, axis=1)
    radial = points / radii[:, None]
    tangent = np.column_stack([-radial[:, 1], radial[:, 0]])
    nr = np.sum(normals * radial, axis=1)
    good = nr < -0.1
    # A generic smooth periodic representation; fixed before any evaluation.
    harmonics = np.arange(1, 7)

    def design(t):
        z = np.asarray(t)[:, None] * harmonics
        return np.column_stack([np.ones(len(t)), np.sin(z), np.cos(z)])

    z = theta[good, None] * harmonics
    deriv = np.column_stack([np.zeros(good.sum()), np.cos(z) * harmonics, -np.sin(z) * harmonics])
    dr = -radii[good] * np.sum(normals[good] * tangent[good], axis=1) / nr[good]
    a = np.vstack([design(theta), 0.2 * deriv])
    b = np.r_[radii, 0.2 * dr]
    penalty = 1e-6 * np.r_[0., harmonics ** 4, harmonics ** 4]
    coef = np.linalg.solve(a.T @ a + np.diag(penalty), a.T @ b)
    rounding = None
    if simplify:
        # Select the shortest decimal representation that preserves contact
        # fit at the supplied simulator's 2e-4 feasibility tolerance. Selection
        # uses ONLY demonstration contacts, never evaluation labels/scores.
        tolerance = max(2e-4, 3 * float(np.sqrt(np.mean((design(theta) @ coef - radii)**2))))
        for decimals in range(7):
            candidate = np.round(coef, decimals)
            if np.max(np.abs(design(theta) @ candidate - radii)) <= tolerance:
                coef = candidate
                rounding = decimals
                break
    grid = make_eval_grid(61, 'wavy')
    if simplify:
        # Evaluate the learned implicit constraint. Skip zero coefficients;
        # these are learned and compressed, not constants from task data.
        angle = np.arctan2(grid[:, 1], grid[:, 0])
        radius = np.full(len(grid), coef[0])
        for j, h in enumerate(harmonics):
            if coef[1+j] != 0:
                radius += coef[1+j] * np.sin(h * angle)
        for j, h in enumerate(harmonics):
            if coef[1+len(harmonics)+j] != 0:
                radius += coef[1+len(harmonics)+j] * np.cos(h * angle)
        pred = (radius**2 - (grid[:, 0]**2 + grid[:, 1]**2) <= 0).astype(int)
    else:
        pred = (np.linalg.norm(grid, axis=1) >= design(np.arctan2(grid[:, 1], grid[:, 0])) @ coef).astype(int)
    result = dict(method='gp_acl', constraint='wavy', predictions=pred.tolist(),
                  n_demos=len(demos), settings=dict(eval_grid=61, seed=seed),
                  audit_variant='better_simplified_boundary_fit' if simplify else 'better_boundary_fit',
                  n_contacts=len(points), fitted_coefficients=coef.tolist(), rounding_decimals=rounding)
    output.write_text(json.dumps(result))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--demos', type=Path)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.variant in ('better_boundary_fit', 'better_simplified_boundary_fit'):
        fit_boundary(json.loads(args.demos.read_text()), args.output, args.seed,
                     simplify=args.variant == 'better_simplified_boundary_fit')
        return
    if args.variant == 'weaker_no_gradients':
        # The derivative observations carry the sign information; remove this
        # key ingredient while keeping all 12 demonstrations and contacts.
        demos = json.loads(args.demos.read_text())
        points = np.vstack([acl.kkt_tight_points(np.asarray(path))[0] for path in demos])
        gp = acl.GPModel(points, np.zeros(len(points)), [('value', -1)] * len(points))
        pred = acl.predict_grid(gp, make_eval_grid(61, 'wavy'))
        args.output.write_text(json.dumps(dict(method='gp_acl', constraint='wavy',
            predictions=pred.tolist(), n_demos=len(demos), settings=dict(eval_grid=61, seed=args.seed),
            audit_variant='weaker_no_gradients')))
        return
    config = acl.build_parser().parse_args([])
    config.seed = args.seed
    config.output = args.output
    if args.variant == 'weaker_no_acquisition':
        config.iterations = 1
    elif args.variant == 'weaker_random':
        config.mode = 'random'
    elif args.variant == 'more_training':
        config.n_features = 1000
    elif args.variant == 'shortcut_over_budget':
        config.iterations = 5
    elif args.variant != 'reference':
        raise ValueError(args.variant)
    demos = []
    original = acl.generate_demo

    def record_demo(*a, **kw):
        path = original(*a, **kw)
        if path is not None:
            demos.append(path.tolist())
        return path

    acl.generate_demo = record_demo
    result = acl.run_experiment(config)
    args.output.write_text(json.dumps(result))
    if args.demos:
        args.demos.write_text(json.dumps(demos))


if __name__ == '__main__':
    main()
