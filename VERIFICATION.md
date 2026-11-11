# Verification

Job `c7324ec1-f1e` exited 0 and passed `python validate.py --post-design`. It ran these checks:
- task-generator gradient and boundary checks for both splits;
- GP likelihood, hinge and tied-prior gradient checks;
- posterior and random-Fourier-feature derivative checks;
- reduced KKT checks;
- Problem 3 crossing-enumeration and trust-region checks;
- rank arithmetic checks;
- expert feasibility and KKT checks.

Hidden test, 36 paired instances per experiment (audit-cohort ranks):

| Method | Rank [95% paired interval] |
|---|---|
| Random | −0.906 [−1.586, −0.230] |
| Fill (better) | +1.088 [0.789, 1.378] |
| Previous reference | −0.683 [−1.332, −0.038] |
| Learned prior | −0.759 [−1.579, +0.069] |
| No acquisition | −2.156 |
| Raw gradients | −4.328 |
| No gradients | fails the success gate |
| P = 2 | +0.151 [−0.409, +0.681] (indistinguishable) |
| Public-formula fit | −0.184 [−1.699, +1.074] |

All 17 adapted interface probes and seven legacy artifacts fail the run. The full audit is in `eval/test/hardening_summary.json`.
