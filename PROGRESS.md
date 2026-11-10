# Progress

Job 3df03bad-304 was cancelled after completing the development reference and random controls and six g3 test reference instances. The reference failed the fixed g3 unsafe-recall gate: 0.577024 < 0.60. Its rank is null; random alone passed and ranked zero. Reference boundary balanced accuracy was 0.704023 (g3) and 0.737435 (stress), versus random 0.700312 and 0.768865. The full development evidence is preserved in eval/test/hardening_sixth_summary.json. No complete sixth-job hidden comparison or new published scores exist.

The changed expert initialization passed independent feasibility, reconstructed dynamics and KKT checks, with maximum representative stationarity residuals below 3.3e-7. Active acquisition obtained more contact observations than random, yet did not recover the required scientific effect. Lowering contact tolerance barely changes the observations. These diagnostics do not support relaxing success gates or substituting normalized gradients for the reference.

Rechecking the paper exposed an outstanding procedural departure: Equation 10 prints Phi(+mean/sd), while the implementation used Phi(-mean/sd) to match the prose's g<=0 safety convention. The printed positive sign is now restored, with a regression check of both endpoint inequalities. The discrepancy between equation and prose is documented. No additional training objective is introduced.

The next combined job fixes every other setting, checks the mathematical primitives and expert, reruns affected reference/random and better/weaker variants on both paired splits, replays fixed shortcuts, and measures paired uncertainty. GP fit diagnostics are recorded as unscored stderr. Successful reference runs alone may replace the score files. Source-versioned checkpoints resume interruptions.

The last complete audit remains eval/test/hardening_fourth_summary.json: all twelve adapted shortcuts and seven legacy artifacts failed, and stronger variants beat the reference beyond paired uncertainty. Random ranked +0.603014 [-0.059525, 1.300636] on test and +0.519918 on train. Those scores remain interim evidence pending the restored-equation audit. The milestone remains method_runs.
