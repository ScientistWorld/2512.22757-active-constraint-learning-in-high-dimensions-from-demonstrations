# Verification

Job `1a6389bd-3dd` exited 0 and passed `python validate.py --post-design`. It ran these checks:
- task-generator gradient and boundary checks for both splits, and matched development/hidden difficulty;
- GP likelihood, hinge and prior-mean gradient checks;
- posterior and random-Fourier-feature derivative checks;
- reduced KKT checks;
- Problem 3 checks;
- rank arithmetic checks;
- expert feasibility and KKT checks.

The reference and the random control were rerun end to end from the final code on both splits and their raw records frozen. The task files and method are byte-identical to that job's.

Hidden test, 72 paired instances per experiment (core cohort; evaluator cohort in HARDENING.md):

| Method | Rank [95% paired interval] |
|---|---|
| Fill + top-up (better) | +0.884 [0.644, 1.146] |
| Fill (better) | +0.728 [0.531, 0.918] |
| Top-up (better) | +0.164 [0.017, 0.335] |
| Previous reference (noise floor 1e-10) | −0.064 [−0.566, 0.446] (tie) |
| P = 2 | −0.390 [−0.663, −0.123] |
| Simple coverage sweep | −1.101 [−1.627, −0.605] |
| Random | −1.269 [−1.784, −0.753] |
| Learned prior | −1.646 [−2.125, −1.168] |
| Family-fit shortcut | −2.725 [−3.175, −2.287] |
| No acquisition | −3.249 [−3.727, −2.787] |
| Raw gradients | −4.562 [−5.135, −3.999] |
| No gradients | fails the success gate |

Paper direction (reference − random, uniform accuracy): g3 +0.0017 [−0.0030, +0.0062]; complex +0.0028 [+0.0003, +0.0052].
