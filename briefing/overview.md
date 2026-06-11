# Overview

This paper studies learning hidden robot safety constraints from demonstrations. Its core result is GP-ACL, which repeatedly learns a GP boundary model from KKT-derived tight points and gradients, samples plausible constraints, and asks for demonstrations that should reduce posterior uncertainty near likely obstacle boundaries.

This workspace implements scaled 2D obstacle-recovery benchmarks that preserve the paper's main computational loop and compare it with the paper's random start/goal query baseline. The main score is safe/unsafe classification accuracy under a fixed demonstration budget.

- **Paper ID:** 2512.22757
- **Title:** Active Constraint Learning in High Dimensions from Demonstrations
- **Domain:** Robot learning / learning from demonstrations
- **TL;DR:** GP-ACL actively chooses demonstrations that reduce uncertainty in a GP constraint model, improving hidden obstacle recovery over random query selection.

## Short Summary

The paper addresses learning unknown robot safety constraints from sparse locally optimal demonstrations. It extracts tight constraint states and gradient information from KKT optimality conditions, fits a GP constraint representation, samples plausible posterior constraints, and chooses new start/goal states near uncertain sampled boundaries. The reported experiments show higher safe/unsafe constraint recovery accuracy than random start/goal sampling across simulated robot systems and hardware.

## Key Results

- On the paper's unicycle complex-constraint setting `g3`, GP-ACL reports 0.9956 accuracy versus 0.9492 for the random sampling baseline.
- This reproduction preserves the GP-ACL computational loop at reduced 2D scale and scores a nonlinear boundary-band recovery task.
- The reproduced scaled benchmark reaches 0.9495 accuracy for GP-ACL versus 0.9245 for the random-query baseline with the same 12 demonstrations.
