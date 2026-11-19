# Final hardening checklist

- [x] Read the paper, REDTEAM.md, REVIEW.md and the previous HARDENING.md before deciding fixes.
- [x] Add the paper's unicycle (g1–g3, 4D, T = 30) and quadrotor (g5, 12D, T = 20) experiments with hidden, evaluator-owned instances and expert.
- [x] Reference: full-dynamics KKT Problems 7–8; dimension-general GP and Problems 3–6.
- [x] Development smoke test and pilot; paper-formula-fit shortcut probe added.
- [x] 3D deformation enlarged (matched across splits) until the published-formula fit no longer outranks the reference.
- [x] Evaluator robustness after a stalled job (stderr drain, spawned workers, compact records).
- [x] Reference and random control rerun from final code on both splits; raw records frozen; scores published.
- [x] Hidden-test audit: better, weaker, simple, shortcut, alternative and interface members.
- [x] HARDENING.md; briefing and scoring documents updated for four experiments.
- [x] scripts/run.sh restored to end-to-end reproduction.
- [x] Final commit and action: done.
