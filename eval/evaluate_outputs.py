"""Evaluate saved constraint-learning predictions.

This module intentionally does not import from method/. It reads artifacts
written by any method that follows the JSON output contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def confusion(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    safe_pred = pred == 1
    safe_true = truth == 1
    tp = np.sum(safe_pred & safe_true)
    tn = np.sum(~safe_pred & ~safe_true)
    fp = np.sum(~safe_pred & safe_true)
    fn = np.sum(safe_pred & ~safe_true)
    total = len(pred)
    return {
        "accuracy": float((tp + tn) / total),
        "false_positive_rate": float(fp / max(1, np.sum(safe_true))),
        "false_negative_rate": float(fn / max(1, np.sum(~safe_true))),
    }


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text())
    for key in ("method", "predictions", "truth"):
        if key not in data:
            raise ValueError(f"{path} is missing required key {key!r}")
    pred = np.asarray(data["predictions"], dtype=int)
    truth = np.asarray(data["truth"], dtype=int)
    if pred.shape != truth.shape:
        raise ValueError(f"{path} predictions and truth shapes differ: {pred.shape} vs {truth.shape}")
    all_metrics = confusion(pred, truth)
    metrics = {
        "accuracy": all_metrics["accuracy"],
        "n_demos": float(data.get("n_demos", 0)),
    }
    return data["method"], metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="scaled_2d_obstacle_recovery")
    parser.add_argument("--active", type=Path, default=Path("outputs/gp_acl_circle.json"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/random_baseline_circle.json"))
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
