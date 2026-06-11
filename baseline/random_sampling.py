"""Random start/goal query baseline used in the paper comparison."""

from __future__ import annotations

import json
from pathlib import Path

from method.gp_acl import build_parser, run_experiment


def main() -> None:
    parser = build_parser()
    parser.set_defaults(mode="random", output=Path("outputs/random_baseline_circle.json"))
    args = parser.parse_args()
    args.mode = "random"
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "method": result["method"], "accuracy": result["history"][-1]["accuracy"]}))


if __name__ == "__main__":
    main()
