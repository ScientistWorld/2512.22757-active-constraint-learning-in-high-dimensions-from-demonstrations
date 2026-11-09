# Progress

Read the paper and REDTEAM.md; restored the paper's KKT, derivative GP, posterior sampling and continuous acquisition programs. Applied the evaluator, privacy, ranking and shortcut fixes together and ran all comparisons in job 5f293a40-e15. Interface and mathematical checks passed; the reference failed the recall gate and active sampling underfilled its demonstration budget. No successful reproduction is claimed.

The next combined job restores physical cost/position units, improves numerical solutions of the same constrained programs, uses multiple starts for marginal likelihood, and reruns the final reference, random control, better/weaker variants and affected shortcuts on both splits. It resumes versioned per-instance checkpoints after preemption. Final scores and paired evidence will be regenerated from that job; HARDENING.md records the failed first verification separately.
