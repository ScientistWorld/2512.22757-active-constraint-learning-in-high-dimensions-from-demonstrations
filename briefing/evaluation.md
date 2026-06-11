# Evaluation

## Metrics

The primary metric is constraint recovery accuracy: the fraction of sampled constraint states whose safe/unsafe label is predicted correctly. Higher is better. The reproduction also records the number of demonstrations used, because the paper's claim is about improving accuracy from sparse but informative demonstrations rather than simply collecting more data.

## Baselines and Targets

The paper compares active querying against a random-sampling baseline. In the closest 2D double-integrator setting reported in Appendix B.2, the active method reaches 0.9952 accuracy on constraint `g1` over 2500 samples, while the random baseline reaches 0.9900. The main text reports larger gaps for unicycle and quadcopter settings.

## Evaluation Protocol

Run both the proposed method and the random-query baseline with the same initial demonstrations, iteration count, start/goal query budget, trajectory horizon, and evaluation grid. Each method writes prediction artifacts under `outputs/`; `scripts/evaluate.sh` reads those artifacts and writes `scoring/scores.json`.
