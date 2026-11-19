# Verification

Job `e87b2f87-237` exited 0 and passed `python validate.py --post-design`.

## Checks run in the job

- Task-generator gradient and boundary checks for both double-integrator splits.
- GP likelihood, hinge and prior-mean gradient checks, in 2D and 3D.
- Posterior and RFF derivative checks.
- KKT checks:
  - double integrator, unicycle and quadrotor;
  - full-dynamics programs agree with the position residuals.
- Problem 3–6 checks (2D and a 3D batch).
- Rank arithmetic checks.
- Expert checks:
  - feasibility;
  - Prob. 1 stationarity ≤ 2.5e-6;
  - lifted unicycle/quadrotor dynamics exact to 4e-15 against independent forward models.
- Deformation-size match between splits (quadrotor RMS 0.52 vs 0.52 on the boundary band).

The reference and the random control were rerun end to end from the final code on both splits and all four experiments, and their raw records frozen. The method and task files are byte-identical to that job's.

## Hidden test, core cohort rank

| Method | Rank [95% paired interval] |
|---|---|
| Fill + top-up (better) | +0.357 [0.118, 0.601] |
| Top-up (better) | +0.180 [0.059, 0.308] |
| Fill (better) | +0.174 [−0.044, 0.378] |
| P = 2 | −0.192 [−0.426, 0.046] |
| Published-formula fit (shortcut, core+1) | −0.257 [−0.652, 0.132] |
| Simple coverage sweep | −0.485 [−0.914, −0.050] |
| Development-generator fit (shortcut, core+1) | −0.842 [−1.023, −0.663] |
| Learned prior | −1.199 [−1.627, −0.775] |
| Random | −1.680 [−2.095, −1.242] |
| Zero prior (core+1) | −3.434 [−3.929, −2.975] |
| No acquisition | −4.162 [−4.576, −3.752] |
| Raw gradients | −6.336 [−6.878, −5.795] |
| No gradients, interface probes | fail the run |

## Paper direction (random − reference, uniform accuracy)

| Experiment | Difference [95% interval] |
|---|---|
| Quadrotor | −0.0030 [−0.0053, −0.0006] |
| Unicycle | +0.0001 [−0.0058, +0.0058] |
| Double integrator g3 | −0.0017 [−0.0061, +0.0031] |
| Complex boundaries | −0.0028 [−0.0052, −0.0003] |
