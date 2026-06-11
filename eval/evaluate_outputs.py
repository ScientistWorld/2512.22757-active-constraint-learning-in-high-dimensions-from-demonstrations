"""Evaluate saved constraint-learning predictions.

This module intentionally does not import from method/. It reads artifacts
written by any method that follows the JSON output contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.constraints import CONSTRAINTS, make_eval_grid


def confusion(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    safe_pred = pred == 1
    safe_true = truth == 1
    tp = np.sum(safe_pred & safe_true)
    tn = np.sum(~safe_pred & ~safe_true)
    fp = np.sum(~safe_pred & safe_true)
    fn = np.sum(safe_pred & ~safe_true)
    total = len(pred)
    safe_total = max(1, np.sum(safe_true))
    unsafe_total = max(1, np.sum(~safe_true))
    safe_recall = float(tp / safe_total)
    unsafe_recall = float(tn / unsafe_total)
    return {
        "accuracy": float((tp + tn) / total),
        "balanced_accuracy": float(0.5 * (safe_recall + unsafe_recall)),
        "safe_recall": safe_recall,
        "unsafe_recall": unsafe_recall,
        "false_positive_rate": float(fp / safe_total),
        "false_negative_rate": float(fn / unsafe_total),
    }


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text())
    for key in ("method", "predictions"):
        if key not in data:
            raise ValueError(f"{path} is missing required key {key!r}")
    pred = np.asarray(data["predictions"], dtype=int)
    constraint_name = data.get("constraint", "wavy")
    if constraint_name not in CONSTRAINTS:
        raise ValueError(f"{path} references unknown constraint {constraint_name!r}")
    eval_grid = int(data.get("settings", {}).get("eval_grid", 61))
    grid = make_eval_grid(eval_grid, constraint_name)
    signed_distance = CONSTRAINTS[constraint_name]["function"](grid)
    truth = (signed_distance <= 0.0).astype(int)
    if pred.shape != truth.shape:
        raise ValueError(f"{path} predictions and truth shapes differ: {pred.shape} vs {truth.shape}")
    all_metrics = confusion(pred, truth)
    near_boundary = np.abs(signed_distance) <= np.quantile(np.abs(signed_distance), 0.5)
    near_boundary_metrics = confusion(pred[near_boundary], truth[near_boundary])
    metrics = {
        "accuracy": all_metrics["accuracy"],
        "balanced_accuracy": all_metrics["balanced_accuracy"],
        "near_boundary_balanced_accuracy": near_boundary_metrics["balanced_accuracy"],
        "safe_recall": all_metrics["safe_recall"],
        "unsafe_recall": all_metrics["unsafe_recall"],
        "n_demos": float(data.get("n_demos", 0)),
    }
    return data["method"], metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="scaled_2d_obstacle_recovery")
    parser.add_argument("--active", type=Path, default=Path("outputs/gp_acl_wavy.json"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/random_baseline_wavy.json"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    results = {}
    for path in (args.active, args.baseline):
        if path.exists():
            name, metrics = load_result(path)
            results[name] = metrics
    if not results:
        raise SystemExit("No result artifacts found to evaluate.")

    scores = {"experiments": {args.experiment: {"results": results}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
