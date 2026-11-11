# Verification

Job `3b3ba72f-7f9` exited 0 and passed `python validate.py --post-design`. It ran these checks:
- task-generator gradient and boundary checks for both splits;
- matched difficulty of the development and hidden stress families (radius RMS 0.1398 vs 0.1410, slope RMS 0.655 vs 0.692, narrowest radius 0.7465 vs 0.7466);
- GP likelihood, hinge and tied/zero prior-mean gradient checks;
- posterior and random-Fourier-feature derivative checks;
- reduced KKT checks;
- Problem 3 checks;
- rank arithmetic checks;
- expert feasibility and KKT checks.

Hidden test, 36 paired instances per experiment (core cohort; evaluator audit cohort in HARDENING.md):

| Method | Rank [95% paired interval] |
|---|---|
| Fill + top-up (better) | +1.331 [1.005, 1.680] |
| Fill (better) | +0.887 [0.623, 1.149] |
| Top-up (better) | +0.314 [0.131, 0.528] |
| P = 2 | +0.194 [−0.290, 0.653] (indistinguishable) |
| Random | −0.777 [−1.347, −0.195] |
| Learned prior | −0.780 [−1.518, −0.051] |
| No acquisition | −1.837 |
| Raw gradients | −4.281 |
| No gradients | fails the success gate |
| Stress-generator fit on stress (shortcut) | +0.539 [0.223, 0.864]; was +1.572 before the private family (three-member cohort) |

`VERIFY_JOB` reran both controls from the final code with no caches and compared them with the frozen records (see HARDENING.md).
