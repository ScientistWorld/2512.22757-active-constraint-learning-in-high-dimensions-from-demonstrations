# Final hardening checklist

- [x] Read the paper, REDTEAM.md, REVIEW.md and the previous HARDENING.md before deciding fixes.
- [x] Test the paper's partial initial set on development (rejected by a pre-fixed rule: Alg. 1 stalls).
- [x] Diagnose what limits uniform accuracy (development pilot): spurious unsafe regions from a singular 1e-10 noise floor; hollow interiors.
- [x] Restore a well-conditioned noise floor (1e-4, GPyTorch's default), chosen on development by a pre-fixed rule.
- [x] Enlarge the hidden test from 36 to 72 instances per experiment (paired noise).
- [x] Rejected-request diagnostic (no expert solver failures); 80-request allowance tried and reverted.
- [x] Reference and random control rerun from final code on both splits; raw records frozen.
- [x] Hidden-test audit: better, weaker, previous-reference, simple, shortcut and alternative members.
- [x] HARDENING.md with before/after evidence; briefing and scoring documents updated (test size, noise floor).
- [x] scripts/run.sh restored to end-to-end reproduction.
- [x] Final commit and action: done.
