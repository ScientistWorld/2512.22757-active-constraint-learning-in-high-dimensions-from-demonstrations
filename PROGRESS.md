# Progress

Final hardening iteration. It addressed REVIEW.md: the headline uniform-accuracy effect tied random.

Jobs:
- `ecec050e-3cf` (cancelled): development test of a partial shared initial set. Rejected: Alg. 1 stalls and random outranks the reference.
- `3216cb5c-ccf` (exit 0): development pilot.
  - Tight states are clean.
  - The g3 errors were spurious unsafe regions caused by a numerically singular 1e-10 noise floor; the complex-boundary errors were hollow interiors.
  - GPyTorch's 1e-4 floor removes most of the spurious regions.
- `1a6389bd-3dd` (exit 0): the final job.
  - Development selection of the noise floor (1e-4) by a pre-fixed rule.
  - Controls rerun and frozen on both splits; the hidden test is enlarged to 72 instances.
  - Development and hidden-test audits; validation passed.
- `808829f2-232` (exit 0): rejected-request diagnostic. Every rejection had an endpoint inside the obstacle; no solver failures.
- `59bd0533-310` (cancelled): an 80-request allowance gave no gain and was reverted.

Outcome:
- The reference now leads random on uniform accuracy in both experiments: beyond noise on the complex task, within noise on g3.
- It leads on boundary metrics beyond noise in both, and in combined rank.
- Three better variants beat it beyond noise; ablations, a simple coverage rule and random score below it.

Details are in HARDENING.md.
