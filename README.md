# Active Constraint Learning in High Dimensions from Demonstrations

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2512.22757"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2512.22757-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2512.22757](https://arxiv.org/abs/2512.22757) |
| **Domain** | Robot learning / learning from demonstrations |
| **TL;DR** | GP-ACL actively chooses demonstrations that reduce uncertainty in a GP constraint model, improving hidden obstacle recovery over random query selection. |

## Summary

The paper addresses learning unknown robot safety constraints from sparse locally optimal demonstrations. It extracts tight constraint states and gradient information from KKT optimality conditions, fits a GP constraint representation, samples plausible posterior constraints, and chooses new start/goal states near uncertain sampled boundaries. The reported experiments show higher safe/unsafe constraint recovery accuracy than random start/goal sampling across simulated robot systems and hardware.

## Problem

Learn an unknown safety constraint in a continuous robot state or configuration space from a small sequence of expert demonstrations. Demonstrations are feasible, locally optimal trajectories between requested start and goal states, but the learner does not directly observe the hidden obstacle or constraint function that shaped them.

## The paper's method

The paper introduces Gaussian Process Active Constraint Learning (GP-ACL), an iterative demonstration-query algorithm for learning unknown continuous safety constraints from locally optimal demonstrations.

## Key results of the paper

- On the paper's unicycle complex-constraint setting `g3`, GP-ACL reports 0.9956 accuracy versus 0.9492 for the random sampling baseline.
- This reproduction preserves the GP-ACL computational loop at reduced 2D scale and scores a nonlinear boundary-band recovery task.
- The reproduced scaled benchmark reaches 0.9495 accuracy for GP-ACL versus 0.9245 for the random-query baseline with the same 12 demonstrations.

## Run it

```bash
pip install sciworld
sw research 2512.22757 --mode innovate --agent claude-code
sw evaluate 2512.22757
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

