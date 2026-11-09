# Gym scoring audit

Results and conclusions: [../REDTEAM.md](../REDTEAM.md). Exact test metrics and normalization: [summary.json](summary.json). Original reference: [saved_reference_scores.json](saved_reference_scores.json).

Run in the gym's prepared container at `/home/user`:

```bash
bash redteam/shortcut_oracle/run.sh
# Or reproduce the complete audit (CPU, roughly minutes rather than hours):
bash redteam/run_all.sh
```

Actual computation belongs in a submitted job, not on the login node. Nothing is installed by the runners. They use the gym's NumPy/SciPy environment. The complete audit ran in two CPU jobs; the second job ran [run_followup.sh](run_followup.sh).

Every case has `run.sh`, the exact proposed `artifact.json`, a `status.json`, and train/test score JSON when that stage succeeded. Rejected predictions have no fabricated score; [rejections.json](rejections.json) retains their error messages. `run.log` is regenerated locally and ignored by git. All shortcuts are named `shortcut_*`; other folders are scientific variants or reference controls.

[harness.py](harness.py) installs one candidate under `outputs/`, calls `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh` with no arguments, and restores scores, outputs and the task module even after an evaluation error. It does not modify evaluation code, packages or the interpreter. Its preserved workspace state is temporary and ignored. Each evaluator is a new process after the method producer exits. Cases were bundled in two fresh job containers; no per-case fresh-container dispatch was used or needed by the file-only payloads.

[research_variant.py](research_variant.py) runs reference/ablation training and the stronger learned boundary fits. [shortcut.py](shortcut.py) contains the shortcut payloads. Reference trajectories and outputs are generated from visible method code and cached under the ignored `cache/` directory. Running a shortcut from a fresh clone regenerates its needed cache; no hidden reference scores or hidden evaluator modules are read by the payload. The saved original scores are used only by [summarize.py](summarize.py) to report comparison deltas.

The harness's `finally` restoration is for workspace hygiene after recording a run. During the actual hidden invocation, the candidate files are in place exactly as produced. The ground-truth replacement shortcut modifies `data/constraints.py`, a future-agent-visible persistent file; no evaluator process or Python environment is tampered with. Standalone scientific variants reuse the shipped `method/`, so their runners target improve mode; shortcut logic itself needs only visible task generators and legitimately generated reference artifacts.

`summary.json` explicitly names its illustrative audit-cohort normalization. It contains nonstandard JSON NaN/Infinity only in the rejected-for-ranking diagnostic section, matching the evaluator's actual permissive output. It must not be interpreted as evidence that downstream ranking accepts those values.
