# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-11 00:00] - none
- Read the paper and appendix notes for the GP-ACL algorithm.
- Implemented a scaled 2D reproduction of the paper's computational loop: constrained smoothness demonstrations, KKT-style tight point and gradient extraction, GP training with derivative observations, posterior-sample-guided active querying, and random-query baseline.
- Added evaluation, scripts, briefing, and scoring reference files.
- Submitted a CPU test job is still required before claiming `method_runs`.

### [2026-06-11 00:00] - method_runs
- Job `1181296e-3cc` ran the proposed method, random baseline, and evaluator end to end.
- The first completed benchmark was too easy and did not support the core claim, so the next run uses a harder nonlinear obstacle.

## Stop Justification
- Not stopped.
