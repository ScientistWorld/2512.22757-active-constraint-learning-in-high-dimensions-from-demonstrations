# Progress

Final hardening iteration. It addressed REVIEW.md: only the 2D double integrator was scored, and the paper's high-dimensional claims were untested.

Done:
- **New experiments.** Two scored experiments were added from the paper's main text: unicycle constraints g1–g3 with 4D unicycle demonstrations (T = 30), and the hourglass g5 with 12D quadrotor demonstrations (T = 20).
  - Hidden instances (rotation, scale, translation, private deformation) live in `eval/test/`.
  - The evaluator solves and lifts the expert demonstrations itself.
- **Reference.** It now solves Problems 7–8 over each system's full dynamics and runs Problems 3–6 in 2D/3D.
- **Weights.** Unicycle 0.35, quadrotor 0.25, double-integrator g3 0.2, complex 0.2.
- **3D deformation.** Enlarged twice (RMS 0.18 → 0.35 → 0.52, matched across splits) until the published-formula fit no longer outranked the reference on the quadrotor.
- **Evaluator robustness.** Producer stderr drained concurrently, spawned workers, compact records.

Jobs:

| Job | Outcome |
|---|---|
| `a9a7ce06-642` | Smoke test |
| `d2baf85e-52c` | Development pilot |
| `92153af0-4dd` | Stalled; cancelled |
| `b2b22cca-e77` | Final, RMS 0.35 |
| `e87b2f87-237` | Final, RMS 0.52; the baseline the gym ships with |

Outcome on the hidden test:
- **Quadrotor:** the reference beats random beyond noise on every metric.
- **Unicycle:** the reference beats random on boundary metrics; uniform accuracy ties.
- **Variants:** two better variants beat the reference beyond noise. Ablations, a simple coverage rule and random score below it.
- **Shortcuts:**
  - the published-formula fit ranks −0.26 [−0.65, +0.13];
  - the development-generator fit ranks −0.84;
  - every interface probe fails.

Details are in HARDENING.md.
