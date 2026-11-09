# Verification

The first combined job (5f293a40-e15) completed derivative-GP/KKT checks, rank arithmetic checks, both split comparisons and every major shortcut replay. Its final validation failed because reference and random control both failed the quality gate, resulting in all-zero published scores. Actual split predictions and raw metrics differed; the validator's identical-score errors were a consequence of these failures.

The next combined job checks analytic posterior/sample derivatives and continuous scalar constrained solves, reruns every changed reference/control/variant in restored physical units, replays affected shortcuts, computes paired uncertainty and regenerates reference-only scores. Source-versioned records now survive preemptions. Final numerical claims remain pending; no core-claim milestone is asserted.
