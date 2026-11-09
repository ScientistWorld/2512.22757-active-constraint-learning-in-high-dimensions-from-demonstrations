"""Rank deltas under the stated rule; actual future cohort scales are unknown."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def main():
    root = Path('/home/user')
    audit = root/'redteam'
    reference = json.loads((root/'scoring/reference.json').read_text())
    experiment = 'scaled_2d_obstacle_recovery'
    spec = reference['experiments'][experiment]
    saved = json.loads((audit/'saved_reference_scores.json').read_text())['experiments'][experiment]['results']['gp_acl']
    cases = {}
    invalid = {}
    for name in (audit/'cases.txt').read_text().splitlines():
        score = audit/name/'scores_test.json'
        status = audit/name/'status.json'
        if not score.exists():
            invalid[name] = json.loads(status.read_text()) if status.exists() else {'status': 'not run'}
            continue
        metrics = json.loads(score.read_text())['experiments'][experiment]['results']['gp_acl']
        if not all(math.isfinite(metrics[k]) for k in spec['metrics']):
            invalid[name] = {'status': 'nonfinite_metrics', 'metrics': metrics}
            continue
        cases[name] = metrics
    # Illustrative audit-cohort normalization. Not an invented claim about the
    # unseen future attempt population. Componentwise dominance is scale-free.
    scales = {k: float(np.std([saved[k]] + [v[k] for v in cases.values()])) for k in spec['metrics']}

    def delta(metrics, anchor=saved):
        return sum(abs(v['coefficient']) * (1 if v['higher_is_better'] else -1) *
                   (metrics[k] - anchor[k]) / scales[k]
                   for k, v in spec['metrics'].items() if scales[k] > 0) * spec['weight']

    result = dict(reference=saved, normalization='population std of finite audited attempts plus saved reference',
                  std=scales, cases={}, invalid=invalid)
    for name, metrics in cases.items():
        signed = [(1 if v['higher_is_better'] else -1)*(metrics[k]-saved[k]) for k,v in spec['metrics'].items()]
        result['cases'][name] = dict(metrics=metrics, normalized_rank_delta=delta(metrics),
                                   weakly_dominates_saved_reference=all(v >= -1e-12 for v in signed),
                                   quality_sum=sum(abs(v['coefficient'])*metrics[k] for k,v in spec['metrics'].items() if k != 'n_demos'))
    seeds = [cases.get(f'reference_seed_{seed}') for seed in (7, 17, 27)]
    if all(seeds):
        result['reference_seed_ranges'] = {k: dict(min=min(s[k] for s in seeds), max=max(s[k] for s in seeds),
            range=max(s[k] for s in seeds)-min(s[k] for s in seeds)) for k in spec['metrics']}
        ranks = [delta(s) for s in seeds]
        result['reference_seed_rank_spread'] = max(ranks)-min(ranks)
    (audit/'summary.json').write_text(json.dumps(result, indent=2))
    for name, case in result['cases'].items():
        print(name, json.dumps(case), flush=True)
    print('INVALID', json.dumps(invalid), flush=True)


if __name__ == '__main__':
    main()
