# Progress

Final hardening iteration (12). Jobs:
- `c4a8d3c2-15e`: failed on an over-strict unit check.
- `c9eb8251-27a`: cancelled. Learned prior plus new Problem 3: most requests were rejected.
- `fcaa3a0a-3f9`: development-only exploration of unspecified settings (prior mean, Problem 3 radius, posterior draws P).
- `c7324ec1-f1e`: final combined audit, exit 0, validation passed.
- A last small job measures the public-formula shortcut on undeformed hidden shapes, for the "before" score.

Outcome:
- The reference beats random beyond paired noise in combined score.
- A genuinely better variant (interior fill) beats the reference beyond noise.
- The previous method and the ablations score below the reference.
- All shortcut probes fail or no longer pay.

Details and numbers are in HARDENING.md.
