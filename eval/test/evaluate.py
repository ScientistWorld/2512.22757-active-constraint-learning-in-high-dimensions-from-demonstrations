"""Held-out test-slice evaluator for saved constraint-learning predictions.

This evaluator is method-agnostic: it reads JSON artifacts produced under the
shared output contract and scores the complementary half of the evaluation grid.
It intentionally does not import from method/ or eval/train/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from data.constraints import CONSTRAINTS, make_eval_grid


SLICE_NAME = "test"
SPLIT_STYLE = "held-out alternating angular sectors of the fixed boundary-band evaluation grid"
TEST_SECTORS = {1, 3, 5, 7}
N_SECTORS = 8
NEAR_BOUNDARY_FRACTION = 0.35


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc


def discover_result_paths(outputs_dir: Path, constraint: str) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(outputs_dir.glob("*.json")):
        data = _load_json(path)
        if "method" not in data or "predictions" not in data:
            continue
        if data.get("constraint", "wavy") != constraint:
            continue
        paths.append(path)
    return paths


def explicit_result_paths(paths: Iterable[Path | None]) -> list[Path]:
    return [path for path in paths if path is not None and path.exists()]


def train_methods(path: Path, experiment: str) -> set[str]:
    if not path.exists():
        return set()
    data = _load_json(path)
    experiments = data.get("experiments", data)
    exp = experiments.get(experiment, {})
    results = exp.get("results", exp) if isinstance(exp, dict) else {}
    if not isinstance(results, dict):
        raise ValueError(f"{path} does not contain a method-results map for {experiment!r}")
    return {name for name in results if not name.startswith("_")}


def split_indices(n_per_axis: int) -> np.ndarray:
    linear = np.arange(n_per_axis * n_per_axis)
    angle_index = linear % n_per_axis
    sectors = np.floor(angle_index * N_SECTORS / n_per_axis).astype(int)
    return np.flatnonzero(np.isin(sectors, sorted(TEST_SECTORS)))


def near_boundary_indices(indices: np.ndarray, n_per_axis: int) -> np.ndarray:
    offset_index = indices // n_per_axis
    center = (n_per_axis - 1) / 2.0
    distance = np.abs(offset_index - center)
    cutoff = max(1.0, (n_per_axis * NEAR_BOUNDARY_FRACTION) / 2.0)
    return indices[distance <= cutoff]


def confusion_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    safe_true = truth == 1
    unsafe_true = truth == 0
    safe_recall = float(np.mean(pred[safe_true] == 1)) if np.any(safe_true) else 0.0
    unsafe_recall = float(np.mean(pred[unsafe_true] == 0)) if np.any(unsafe_true) else 0.0
    return {
        "accuracy": float(np.mean(pred == truth)),
        "balanced_accuracy": float(0.5 * (safe_recall + unsafe_recall)),
        "safe_recall": safe_recall,
        "unsafe_recall": unsafe_recall,
    }


def load_result(path: Path) -> tuple[str, dict[str, float]]:
    data = _load_json(path)
    for key in ("method", "predictions"):
        if key not in data:
            raise ValueError(f"{path} is missing required key {key!r}")

    constraint_name = data.get("constraint", "wavy")
    if constraint_name not in CONSTRAINTS:
        raise ValueError(f"{path} references unknown constraint {constraint_name!r}")

    eval_grid = _as_int(data.get("settings", {}).get("eval_grid"), 61)
    grid = make_eval_grid(eval_grid, constraint_name)
    truth = (CONSTRAINTS[constraint_name]["function"](grid) <= 0.0).astype(int)
    pred = np.asarray(data["predictions"], dtype=int)
    if pred.shape != truth.shape:
        raise ValueError(f"{path} predictions and truth shapes differ: {pred.shape} vs {truth.shape}")

    indices = split_indices(eval_grid)
    metrics = confusion_metrics(pred[indices], truth[indices])
    near_indices = near_boundary_indices(indices, eval_grid)
    near_metrics = confusion_metrics(pred[near_indices], truth[near_indices])
    metrics["near_boundary_balanced_accuracy"] = near_metrics["balanced_accuracy"]
    metrics["n_demos"] = float(data.get("n_demos", 0))
    return str(data["method"]), metrics


def collect_results(paths: list[Path], required_methods: set[str]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    sources: dict[str, Path] = {}
    for path in paths:
        name, metrics = load_result(path)
        if required_methods and name not in required_methods:
            continue
        if name in results:
            raise ValueError(f"Duplicate method name {name!r} in {sources[name]} and {path}")
        results[name] = metrics
        sources[name] = path

    missing = required_methods.difference(results)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(f"Missing held-out artifacts for method(s) listed in train scores: {missing_list}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method_and_run_name", nargs="?", help="Accepted for compatibility; all train-scored methods are scored by default.")
    parser.add_argument("checkpoint", nargs="?", help="Accepted for compatibility; evaluation reads prediction JSON artifacts.")
    parser.add_argument("--experiment", default="scaled_2d_obstacle_recovery")
    parser.add_argument("--constraint", default="wavy", choices=sorted(CONSTRAINTS))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--active", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_test.json"))
    args = parser.parse_args()

    required_methods = train_methods(args.train_scores, args.experiment)
    paths = explicit_result_paths((args.active, args.baseline))
    if not paths:
        paths = discover_result_paths(args.outputs_dir, args.constraint)
    if not paths:
        raise SystemExit(f"No result artifacts found in {args.outputs_dir} for constraint {args.constraint!r}.")

    results = collect_results(paths, required_methods)
    if not results:
        raise SystemExit("No method artifacts matched the held-out test selection.")

    scores = {
        "_slice": SLICE_NAME,
        "experiments": {
            args.experiment: {
                "slice": SLICE_NAME,
                "split": SPLIT_STYLE,
                "results": results,
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
