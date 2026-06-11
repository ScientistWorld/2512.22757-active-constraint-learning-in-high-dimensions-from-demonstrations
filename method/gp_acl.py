"""Small-scale reproduction of Gaussian-process Active Constraint Learning.

This implements the paper's computation at a reduced scale:
1. generate locally optimal demonstrations for a smoothness objective with an
   unknown obstacle constraint;
2. extract tight points and constraint gradients from KKT stationarity residuals;
3. train an RBF GP on zero-valued tight points and derivative observations;
4. sample posterior-like constraint functions with random Fourier features;
5. query new start/goal states using maximally uncertain sample boundaries and
   perpendicular/parallel search directions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from data.constraints import CONSTRAINTS, default_initial_pairs, make_eval_grid, make_query_grid


@dataclass
class GPModel:
    x_train: np.ndarray
    y_train: np.ndarray
    obs_types: list[tuple[str, int]]
    lengthscale: float = 0.55
    signal: float = 1.0
    noise: float = 2e-4

    def __post_init__(self) -> None:
        k = self._obs_cov(self.x_train, self.x_train)
        k.flat[:: k.shape[0] + 1] += self.noise
        self.alpha = np.linalg.solve(k, self.y_train)

    def _rbf(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        diff = xa[:, None, :] - xb[None, :, :]
        sq = np.sum(diff**2, axis=2)
        return self.signal**2 * np.exp(-0.5 * sq / self.lengthscale**2)

    def _cov_obs_func(self, x_obs: np.ndarray, x_query: np.ndarray, typ: tuple[str, int]) -> np.ndarray:
        base = self._rbf(x_obs, x_query)
        if typ[0] == "value":
            return base
        dim = typ[1]
        return base * (x_query[None, :, dim] - x_obs[:, None, dim]) / self.lengthscale**2

    def _obs_cov(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        out = np.empty((len(self.obs_types), len(self.obs_types)), dtype=float)
        base = self._rbf(xa, xb)
        for i, ti in enumerate(self.obs_types):
            for j, tj in enumerate(self.obs_types):
                if ti[0] == "value" and tj[0] == "value":
                    out[i, j] = base[i, j]
                elif ti[0] == "deriv" and tj[0] == "value":
                    d = ti[1]
                    out[i, j] = base[i, j] * (xb[j, d] - xa[i, d]) / self.lengthscale**2
                elif ti[0] == "value" and tj[0] == "deriv":
                    d = tj[1]
                    out[i, j] = base[i, j] * (xa[i, d] - xb[j, d]) / self.lengthscale**2
                else:
                    di, dj = ti[1], tj[1]
                    delta = 1.0 if di == dj else 0.0
                    diff_i = xa[i, di] - xb[j, di]
                    diff_j = xa[i, dj] - xb[j, dj]
                    out[i, j] = base[i, j] * (delta / self.lengthscale**2 - diff_i * diff_j / self.lengthscale**4)
        return out

    def mean_var(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xq = np.atleast_2d(x_query)
        k_star = np.vstack([self._cov_obs_func(self.x_train[i : i + 1], xq, typ) for i, typ in enumerate(self.obs_types)])
        mean = k_star.T @ self.alpha
        v = np.linalg.solve(self._obs_cov(self.x_train, self.x_train) + np.eye(len(self.obs_types)) * self.noise, k_star)
        var = np.maximum(self.signal**2 - np.sum(k_star * v, axis=0), 1e-8)
        return mean, var

    def grad_mean(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.zeros(2)
        for d in range(2):
            step = np.zeros(2)
            step[d] = eps
            grad[d] = (self.mean_var(x + step)[0][0] - self.mean_var(x - step)[0][0]) / (2 * eps)
        return grad

    def sample_values(self, x_query: np.ndarray, rng: np.random.Generator, n_features: int, alpha: float) -> np.ndarray:
        sample = self.sample_function(rng, n_features=n_features, alpha=alpha)
        return sample(x_query)

    def sample_function(self, rng: np.random.Generator, n_features: int, alpha: float):
        omega = rng.normal(0.0, 1.0 / self.lengthscale, size=(n_features, 2))
        phase = rng.uniform(0.0, 2 * np.pi, size=n_features)
        weights = rng.normal(0.0, 1.0, size=n_features)

        def evaluate(x_query: np.ndarray) -> np.ndarray:
            xq = np.atleast_2d(x_query)
            mean, _ = self.mean_var(xq)
            features = np.sqrt(2.0 / n_features) * np.cos(xq @ omega.T + phase)
            return mean + alpha * (features @ weights)

        return evaluate


def smoothness_objective(flat: np.ndarray, start: np.ndarray, goal: np.ndarray, horizon: int) -> float:
    pts = unpack_path(flat, start, goal, horizon)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(diffs**2))


def unpack_path(flat: np.ndarray, start: np.ndarray, goal: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 2:
        return np.vstack([start, goal])
    return np.vstack([start, flat.reshape(horizon - 2, 2), goal])


def generate_demo(start: np.ndarray, goal: np.ndarray, g_func, horizon: int, rng: np.random.Generator) -> np.ndarray | None:
    if np.any(g_func(np.vstack([start, goal])) > 0.0):
        return None
    line = np.linspace(start, goal, horizon)
    angle = rng.uniform(0, 2 * np.pi)
    bow = np.column_stack([np.sin(np.linspace(0, np.pi, horizon)) * np.cos(angle), np.sin(np.linspace(0, np.pi, horizon)) * np.sin(angle)])
    best = None
    for scale in [0.0, 0.35, -0.35, 0.70, -0.70, 1.05, -1.05]:
        init = (line + scale * bow)[1:-1].ravel()
        cons = [{"type": "ineq", "fun": lambda z, t=t: -g_func(unpack_path(z, start, goal, horizon)[t : t + 1])[0]} for t in range(horizon)]
        res = minimize(
            smoothness_objective,
            init,
            args=(start, goal, horizon),
            method="SLSQP",
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-9, "disp": False},
        )
        if res.success:
            path = unpack_path(res.x, start, goal, horizon)
            if np.max(g_func(path)) <= 2e-4 and (best is None or res.fun < best[0]):
                best = (res.fun, path)
    return None if best is None else best[1]


def kkt_tight_points(path: np.ndarray, residual_quantile: float = 0.70) -> tuple[np.ndarray, np.ndarray]:
    grad_cost = 2.0 * (2.0 * path[1:-1] - path[:-2] - path[2:])
    norms = np.linalg.norm(grad_cost, axis=1)
    if norms.max() < 1e-5:
        return np.empty((0, 2)), np.empty((0, 2))
    threshold = max(np.quantile(norms, residual_quantile), 1e-3)
    idx = np.where(norms >= threshold)[0]
    if len(idx) > 4:
        idx = idx[np.argsort(norms[idx])[-4:]]
    points = path[1:-1][idx]
    grads = -grad_cost[idx]
    grads /= np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-8)
    return points, grads


def train_gp(tight_points: np.ndarray, tight_grads: np.ndarray) -> GPModel:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    obs: list[tuple[str, int]] = []
    for p, g in zip(tight_points, tight_grads):
        xs.append(p)
        ys.append(0.0)
        obs.append(("value", -1))
        for d in range(2):
            xs.append(p)
            ys.append(float(g[d]))
            obs.append(("deriv", d))
    return GPModel(np.asarray(xs), np.asarray(ys), obs)


def predict_grid(gp: GPModel, grid: np.ndarray) -> np.ndarray:
    mean, _ = gp.mean_var(grid)
    return (mean <= 0.0).astype(int)


def active_queries(gp: GPModel, rng: np.random.Generator, n_pairs: int, n_features: int, alpha: float, delta: float) -> list[tuple[np.ndarray, np.ndarray]]:
    grid = make_query_grid(75)
    mean, var = gp.mean_var(grid)
    pairs = []
    for _ in range(max(2, n_pairs)):
        sample_fn = gp.sample_function(rng, n_features=n_features, alpha=alpha)
        sample = sample_fn(grid)
        boundary = np.where(np.abs(sample) <= np.quantile(np.abs(sample), 0.12))[0]
        boundary = boundary[mean[boundary] <= np.quantile(mean, 0.8)] if len(boundary) else np.arange(len(grid))
        k_mi = grid[boundary[np.argmax(var[boundary])]]
        grad = finite_difference_grad(sample_fn, k_mi)
        norm = np.linalg.norm(grad)
        if norm < 1e-6:
            grad = gp.grad_mean(k_mi)
            norm = np.linalg.norm(grad)
        if norm < 1e-6:
            continue
        grad = grad / norm
        perp = np.array([-grad[1], grad[0]])
        for direction in (perp, grad):
            center = k_mi + 0.12 * grad
            if sample_fn(center)[0] <= 0.0:
                center = k_mi - 0.12 * grad
            safe = []
            for sign in (-1.0, 1.0):
                chosen = None
                for tau in np.linspace(0.35, 2.6, 90):
                    cand = np.clip(center + sign * tau * direction, -1.95, 1.95)
                    m, v = gp.mean_var(cand)
                    prob_safe = ndtr((-m[0]) / np.sqrt(v[0]))
                    if sample_fn(cand)[0] <= -delta and prob_safe >= 0.30:
                        chosen = cand
                if chosen is None:
                    chosen = np.clip(center + sign * 2.1 * direction, -1.95, 1.95)
                safe.append(chosen)
            if len(safe) == 2:
                segment = np.linspace(safe[0], safe[1], 25)
                if np.max(sample_fn(segment)) > -delta:
                    pairs.append((safe[0], safe[1]))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def finite_difference_grad(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros(2)
    for d in range(2):
        step = np.zeros(2)
        step[d] = eps
        grad[d] = (func(x + step)[0] - func(x - step)[0]) / (2 * eps)
    return grad


def random_safe_pairs(rng: np.random.Generator, g_func, n_pairs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    while len(pairs) < n_pairs:
        pts = rng.uniform(-1.8, 1.8, size=(2, 2))
        if np.all(g_func(pts) <= 0.0) and np.linalg.norm(pts[0] - pts[1]) > 1.5:
            pairs.append((pts[0], pts[1]))
    return pairs


def run_experiment(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    spec = CONSTRAINTS[args.constraint]
    g_func = spec["function"]
    demos = []
    for s, g in default_initial_pairs():
        path = generate_demo(np.array(s), np.array(g), g_func, args.horizon, rng)
        if path is not None:
            demos.append(path)

    method_name = "gp_acl" if args.mode == "active" else "random_baseline"
    history = []
    for iteration in range(args.iterations):
        tight_points, tight_grads = [], []
        for path in demos:
            pts, grads = kkt_tight_points(path)
            if len(pts):
                tight_points.append(pts)
                tight_grads.append(grads)
        tp = np.vstack(tight_points)
        tg = np.vstack(tight_grads)
        gp = train_gp(tp, tg)
        grid = make_eval_grid(args.eval_grid, args.constraint)
        pred = predict_grid(gp, grid)
        true = (g_func(grid) <= 0.0).astype(int)
        entry = {"iteration": iteration, "accuracy": float(np.mean(pred == true)), "n_demos": len(demos), "n_tight": int(len(tp))}
        if iteration == args.iterations - 1:
            history.append(entry)
            break
        if args.mode == "active":
            pairs = active_queries(gp, rng, args.pairs_per_iter, args.n_features, args.alpha, args.delta)
        else:
            pairs = random_safe_pairs(rng, g_func, args.pairs_per_iter)
        accepted = 0
        for start, goal in pairs:
            path = generate_demo(start, goal, g_func, args.horizon, rng)
            if path is not None:
                demos.append(path)
                accepted += 1
        entry.update({"n_pairs_proposed": len(pairs), "n_pairs_accepted": accepted})
        history.append(entry)

    mean, var = gp.mean_var(grid)
    pred = (mean <= 0.0).astype(int)
    true = (g_func(grid) <= 0.0).astype(int)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    return {
        "method": method_name,
        "constraint": args.constraint,
        "predictions": pred.tolist(),
        "truth": true.tolist(),
        "posterior_mean": mean.tolist(),
        "posterior_var": var.tolist(),
        "history": history,
        "n_demos": len(demos),
        "n_tight": int(len(tp)),
        "settings": settings,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["active", "random"], default="active")
    p.add_argument("--constraint", choices=sorted(CONSTRAINTS), default="wavy")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--pairs-per-iter", type=int, default=4)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--eval-grid", type=int, default=61)
    p.add_argument("--n-features", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("outputs/gp_acl_wavy.json"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_experiment(args)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "method": result["method"], "accuracy": result["history"][-1]["accuracy"]}))


if __name__ == "__main__":
    main()
