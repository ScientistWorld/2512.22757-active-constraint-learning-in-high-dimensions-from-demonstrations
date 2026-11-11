# Final hardening checklist

- [x] Read the paper, REDTEAM.md and the previous HARDENING.md before deciding fixes.
- [x] Restore the paper's zero-mean posterior as the reference candidate; select on development by a pre-fixed rule (rejected: uniform accuracy 0.73).
- [x] Private hidden complex-boundary family at matched difficulty (closes hand-coding the visible generator at the boundary).
- [x] New probe: MAP fit of the visible stress generator; before (development) and after (hidden test).
- [x] Coverage top-up and fill + top-up headroom variants.
- [x] One combined audit job (`3b3ba72f-7f9`).
- [x] Reference-only scores regenerated from final code; fresh-clone reproduction checked.
- [x] HARDENING.md with before/after evidence.
- [ ] scripts/run.sh restored to end-to-end reproduction.
- [ ] Final commit and action: done.
