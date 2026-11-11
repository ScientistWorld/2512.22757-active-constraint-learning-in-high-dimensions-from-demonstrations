# Progress

Final hardening iteration. Jobs:
- `3b3ba72f-7f9` (exit 0): the combined audit.
  - Development-only test of the paper's zero-mean prior: rejected by the pre-fixed rule.
  - Controls on both splits from final code, frozen.
  - Development and hidden-test audits: stress-family shortcut before/after; fill, top-up and fill+top-up variants; ablations; the zero-mean alternative.
- `5aa244fe-ed9` (exit 0): fresh-clone end-to-end reproduction of the reference and random control from the final code. It reproduced every frozen record exactly; validation passed.

Outcome:
- The reference (unchanged method, tied prior) beats random beyond paired noise.
- Three better variants beat the reference beyond noise; the best is fill + top-up, +1.33 [0.99, 1.67] in the core cohort.
- The ablations, learned prior and random control score below it.
- Hand-coding the visible stress generator fell from +1.57 to +0.34 [−0.24, 0.93]. It now loses near-boundary accuracy.

Details and numbers are in HARDENING.md.
