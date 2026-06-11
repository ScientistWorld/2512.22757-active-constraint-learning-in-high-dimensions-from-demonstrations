# Evaluation

## Metrics

The primary metric is constraint recovery accuracy: the fraction of sampled constraint states whose safe/unsafe label is predicted correctly. Higher is better. The reproduction also records the number of demonstrations used, because the paper's claim is about improving accuracy from sparse but informative demonstrations rather than simply collecting more data.

## Baselines and Targets

The paper compares informative demonstration selection against a random-sampling baseline. In the main unicycle setting for the complex constraint `g3`, the reported method reaches 0.9956 accuracy over 2500 sampled constraint states, while the random baseline reaches 0.9492.

## Evaluation Protocol

Run candidate systems and the random-query baseline with the same initial demonstrations, iteration count, start/goal query budget, trajectory horizon, and boundary-band evaluation grid. The boundary-band grid samples radius offsets from -0.12 to +0.12 around the benchmark obstacle boundary, emphasizing the region where classification is most sensitive. Systems may use the separate broad unlabeled candidate grid for choosing demonstrations, but only final prediction artifacts are scored on the boundary band. Each method writes prediction artifacts under `outputs/`; `scripts/evaluate.sh` reads those artifacts and writes `scoring/scores.json`.

The evaluator reads method predictions and metadata from the artifacts, but it recomputes the fixed evaluation grid and safe/unsafe labels from `data/constraints.py`. It does not trust method-written labels.
