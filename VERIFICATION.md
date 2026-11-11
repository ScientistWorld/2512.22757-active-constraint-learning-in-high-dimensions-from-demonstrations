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
| Fill + top-up (better) | +1.331 [0.992, 1.673] |
| Fill (better) | +0.887 [0.631, 1.142] |
| Top-up (better) | +0.314 [0.136, 0.535] |
| P = 2 | +0.194 [−0.312, 0.676] (indistinguishable) |
| Random | −0.777 [−1.368, −0.192] |
| Learned prior | −0.780 [−1.507, −0.046] |
| No acquisition | −1.837 |
| Raw gradients | −4.281 |
| No gradients | fails the success gate |
| Stress-generator fit on stress (shortcut) | +0.539 [0.219, 0.873]; was +1.572 before the private family (three-member cohort) |

Job `5aa244fe-ed9` reran `scripts/reproduce.sh` from the final code with no oracle, run or fit caches. Every frozen record of the reference and random control was reproduced exactly on both splits, with identical predictions and oracle ledgers on all instances. `validate.py --post-design` passed.
