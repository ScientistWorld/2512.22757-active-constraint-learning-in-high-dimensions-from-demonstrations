"""Summarize the final audit from evaluator score files only (no private data).

Inputs: eval/test/hardening_scores.json (hidden-test audit cohort, every
member scored by the unchanged evaluator against the frozen reference) and
eval/train/hardening_scores.json (development audit). Every comparison pairs
identical instances, initial demonstrations and evaluation states.

Ranks follow the evaluator's rule: sum over experiments and metrics of
weight * |coefficient| * better-direction improvement over the reference,
divided by the metric's population standard deviation across a cohort.
  core cohort: the reference, the random control and the honest variants;
  core+X: the core cohort plus one extra member X (shortcuts, probes, alt).
Intervals are 95% paired bootstrap intervals over instances (scales fixed).
Writes eval/test/hardening_summary.json.
"""
import json
from pathlib import Path
import numpy as np

CONFIG = json.loads(Path('eval/test/scoring_config.json').read_text())['experiments']
CORE = ['gp_acl', 'random_baseline', 'better_fill_topup', 'better_fill', 'better_topup',
        'weaker_no_acquisition', 'weaker_two_draws', 'weaker_raw_gradients', 'weaker_learned_prior',
        'simple_coverage_sweep']
METRICS = ['accuracy', 'balanced_accuracy', 'near_boundary_balanced_accuracy', 'safe_recall', 'unsafe_recall']
rng = np.random.default_rng(872)


def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def per_instance(scores, name, exp, metric):
    rows = scores['_paired_metrics'].get(name, {}).get(exp, [])
    return np.array([r[metric] for r in rows])


def cohort_scales(scores, cohort):
    scales = {}
    for exp, spec in CONFIG.items():
        for metric in spec['metrics']:
            values = [scores['experiments'][exp]['results'][n][metric] for n in cohort]
            scales[(exp, metric)] = float(np.std(values))
    return scales


def rank_with_interval(scores, name, scales, draws=4000):
    total = 0.; boot = np.zeros(draws); per_exp = {}
    for exp, spec in CONFIG.items():
        count = len(per_instance(scores, 'gp_acl', exp, 'accuracy'))
        index = rng.integers(0, count, (draws, count))
        contribution = 0.; boot_exp = np.zeros(draws)
        for metric, definition in spec['metrics'].items():
            scale = scales[(exp, metric)]
            if scale == 0: continue
            diff = per_instance(scores, name, exp, metric)-per_instance(scores, 'gp_acl', exp, metric)
            factor = spec['weight']*abs(definition['coefficient'])*(1 if definition['higher_is_better'] else -1)/scale
            contribution += factor*diff.mean(); boot_exp += factor*diff[index].mean(axis=1)
        per_exp[exp] = round(float(contribution), 4); total += contribution; boot += boot_exp
    return [round(float(total), 4), *[round(float(x), 4) for x in np.quantile(boot, [.025, .975])]], per_exp


def paired(scores, name, exp, metric, anchor='gp_acl', draws=10000):
    diff = per_instance(scores, name, exp, metric)-per_instance(scores, anchor, exp, metric)
    boot = diff[rng.integers(0, len(diff), (draws, len(diff)))].mean(axis=1)
    return [round(float(diff.mean()), 4), *[round(float(x), 4) for x in np.quantile(boot, [.025, .975])]]


def summarize(scores, label):
    out = {'failures': {k: v[:2] for k, v in scores['_failures'].items()},
           'evaluator_cohort_rank': {k: (round(v, 4) if v is not None else None)
                                     for k, v in scores['_rank']['values'].items()}}
    passing = [n for n in scores['experiments'][next(iter(CONFIG))]['results'] if n not in scores['_failures']]
    core = [n for n in CORE if n in passing]
    scales = cohort_scales(scores, core)
    out['core_cohort'] = core
    out['core_cohort_rank'] = {}; out['core_rank_by_experiment'] = {}
    for name in core:
        out['core_cohort_rank'][name], out['core_rank_by_experiment'][name] = rank_with_interval(scores, name, scales)
    out['core_plus_one_rank'] = {}
    for name in passing:
        if name in core: continue
        extra_scales = cohort_scales(scores, core+[name])
        out['core_plus_one_rank'][name], out['core_rank_by_experiment'][name] = rank_with_interval(scores, name, extra_scales)
    out['means'] = {exp: {n: {m: round(v[m], 4) for m in METRICS} for n, v in scores['experiments'][exp]['results'].items()
                          if n in passing} for exp in CONFIG}
    out['paired_minus_reference'] = {exp: {n: {m: paired(scores, n, exp, m) for m in METRICS}
                                           for n in passing if n != 'gp_acl'} for exp in CONFIG}
    exp = 'unicycle_paper_constraints'
    out['unicycle_by_constraint'] = {}
    for k, family in enumerate(('g1', 'g2', 'g3')):
        block = {}
        for name in ('gp_acl', 'random_baseline'):
            block[name] = {m: round(float(per_instance(scores, name, exp, m)[k::3].mean()), 4) for m in METRICS}
        diff = per_instance(scores, 'random_baseline', exp, 'accuracy')[k::3]-per_instance(scores, 'gp_acl', exp, 'accuracy')[k::3]
        boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
        block['random_minus_reference_accuracy'] = [round(float(diff.mean()), 4), *[round(float(x), 4) for x in np.quantile(boot, [.025, .975])]]
        out['unicycle_by_constraint'][family] = block
    print(f'===== {label}', flush=True)
    print('failures', out['failures'])
    print('core cohort rank [mean, 95% interval]:')
    for name, value in sorted(out['core_cohort_rank'].items(), key=lambda kv: -kv[1][0]):
        print('  %-28s %s  by experiment %s' % (name, value, out['core_rank_by_experiment'][name]))
    print('core cohort plus one member:')
    for name, value in sorted(out['core_plus_one_rank'].items(), key=lambda kv: -kv[1][0]):
        print('  %-28s %s  by experiment %s' % (name, value, out['core_rank_by_experiment'][name]))
    for exp in CONFIG:
        print('--', exp)
        for name, values in out['means'][exp].items():
            print('  %-28s %s' % (name, values))
        for name in ('random_baseline', 'better_fill_topup', 'better_fill', 'better_topup', 'shortcut_paper_fit'):
            if name in out['paired_minus_reference'][exp]:
                print('  %-28s minus reference %s' % (name, {m: out['paired_minus_reference'][exp][name][m] for m in METRICS[:3]}))
    print('unicycle by constraint', json.dumps(out['unicycle_by_constraint']))
    return out


summary = {}
for side, path in (('test', 'eval/test/hardening_scores.json'), ('development', 'eval/train/hardening_scores.json')):
    scores = read(path)
    if scores is not None and 'gp_acl' in scores['_paired_metrics']:
        summary[side] = summarize(scores, side)
Path('eval/test/hardening_summary.json').write_text(json.dumps(summary, indent=1)+'\n')
