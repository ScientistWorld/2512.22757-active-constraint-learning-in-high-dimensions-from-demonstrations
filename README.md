# Overview

This paper studies learning hidden robot safety constraints from demonstrations. Its core result is GP-ACL, which repeatedly learns a GP boundary model from KKT-derived tight points and gradients, samples plausible constraints, and asks for demonstrations that should reduce posterior uncertainty near likely obstacle boundaries.

This workspace implements a scaled 2D obstacle-recovery benchmark that preserves the paper's main computational loop and compares it with the paper's random start/goal query baseline. The main score is safe/unsafe classification accuracy under a fixed demonstration budget.

- **Paper ID:** <!-- e.g. 1706.03762 -->
- **Title:** <!-- e.g. Attention Is All You Need -->
- **Domain:** <!-- e.g. Computer Vision / Object Detection -->
- **TL;DR:** <!-- One sentence: what this paper does and why it matters -->

## Short Summary

<!-- 3-5 sentences covering the problem, approach, and main findings -->

## Key Results

<!-- 2-3 bullet points: main findings or numbers in bullet points -->

---

# Problem Definition

## Research Question

Learn an unknown safety constraint in a continuous robot state or configuration space from a small sequence of expert demonstrations. Demonstrations are feasible, locally optimal trajectories between requested start and goal states, but the learner does not directly observe the hidden obstacle or constraint function that shaped them.

## Why It Matters

Robots deployed around unknown obstacles or task restrictions need to infer safety boundaries from limited expert behavior. A good learner should recover the safe/unsafe split accurately while requiring few demonstrations, because demonstrations may be expensive or risky to collect.

## Success Criteria

A successful system asks for informative demonstrations, fits a constraint representation from the resulting trajectories, and correctly classifies held-out constraint states as safe or unsafe. It should improve over a random start/goal query strategy under the same demonstration budget.

---

# Paper's Method

## Key Contribution

The paper introduces Gaussian Process Active Constraint Learning (GP-ACL), an iterative demonstration-query algorithm for learning unknown continuous safety constraints from locally optimal demonstrations.

## Approach

At each iteration, the learner extracts tight states and constraint-gradient estimates from demonstrations using KKT optimality conditions. These zero-valued boundary observations and gradient observations train a Gaussian-process representation of the unknown constraint. The learner samples plausible constraint functions from the posterior, finds high-uncertainty points on sampled boundaries, and generates start/goal pairs in directions parallel and orthogonal to the sampled boundary gradient. New demonstrations from those start/goal pairs are added to the dataset.

## Main Claims

The primary claim is that actively selected demonstrations recover unknown nonlinear constraints more accurately than random start/goal selection with the same sparse demonstration budget. Secondary claims are that the method applies to high-dimensional nonlinear dynamics and can handle both simulation and robot hardware settings.

---

# Evaluation

## Metrics

The primary metric is constraint recovery accuracy: the fraction of sampled constraint states whose safe/unsafe label is predicted correctly. Higher is better. The reproduction also records the number of demonstrations used, because the paper's claim is about improving accuracy from sparse but informative demonstrations rather than simply collecting more data.

## Baselines and Targets

The paper compares active querying against a random-sampling baseline. In the main unicycle setting for the complex constraint `g3`, the active method reaches 0.9956 accuracy over 2500 sampled constraint states, while the random baseline reaches 0.9492.

## Evaluation Protocol

Run both the proposed method and the random-query baseline with the same initial demonstrations, iteration count, start/goal query budget, trajectory horizon, and boundary-band evaluation grid. The boundary-band grid samples radius offsets from -0.12 to +0.12 around the benchmark obstacle boundary, emphasizing the region where the paper reports most false-positive and false-negative errors. The method's active query search uses a separate broad unlabeled candidate grid; only the final prediction artifacts are scored on the boundary band. Each method writes prediction artifacts under `outputs/`; `scripts/evaluate.sh` reads those artifacts and writes `scoring/scores.json`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 19m | **GPU**: 0.0h
- **Jobs**: 7 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-11 00:00] - none
- Read the paper and appendix notes for the GP-ACL algorithm.
- Implemented a scaled 2D reproduction of the paper's computational loop: constrained smoothness demonstrations, KKT-style tight point and gradient extraction, GP training with derivative observations, posterior-sample-guided active querying, and random-query baseline.
- Added evaluation, scripts, briefing, and scoring reference files.
- Submitted a CPU test job is still required before claiming `method_runs`.

### [2026-06-11 00:00] - method_runs
- Job `1181296e-3cc` ran the proposed method, random baseline, and evaluator end to end.
- The first completed benchmark was too easy and did not support the core claim, so the next run uses a harder nonlinear obstacle.

### [2026-06-11 16:28] - core_claim
- Job `1f14600a-6d2` ran the narrowed boundary-band wavy obstacle benchmark end to end.
- GP-ACL outperformed the random-query baseline with the same 12-demonstration budget: 0.9492 versus 0.9371 accuracy.
- The random baseline stayed below the 95% non-triviality threshold, so the scaled benchmark can distinguish active constraint learning from random sampling.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
