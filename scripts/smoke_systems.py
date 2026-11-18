"""Development smoke test of the unicycle/quadrotor experiments (job only).

Runs a few development instances per new experiment for the given manifests,
in parallel, and prints per-instance metrics and wall time. Uses the
development evaluator's own run/metric code; nothing is frozen or published.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'eval'/'train'))
import evaluate as ev  # noqa: E402

MANIFESTS = {
    'reference': ['python', 'method/gp_acl.py', '--variant', 'reference', '--prior', 'tied', '--noise-floor', '0.0001'],
    'random': ['python', 'method/gp_acl.py', '--variant', 'random', '--prior', 'tied', '--noise-floor', '0.0001'],
    'topup': ['python', 'method/gp_acl.py', '--variant', 'fill_topup', '--prior', 'tied', '--noise-floor', '0.0001'],
}


def one(job):
    name, exp, index = job
    start = time.time()
    task = ev.make_task(exp, index)
    record = ev.run_instance({'protocol': 'acl-v2', 'command': MANIFESTS[name]}, task)
    seconds = time.time()-start
    out = {'method': name, 'experiment': exp, 'index': index, 'family': getattr(task, 'family', ''),
           'seconds': round(seconds, 1), 'failure': record['failure'],
           'demos': len(record['initial_demos'])+sum(x['demo'] is not None for x in record['ledger']),
           'requests': len(record['ledger'])}
    if not record['failure']:
        points, truth, uniform = task.evaluation(); p = np.asarray(record['predictions']); band = ~uniform
        near = band & (task.boundary_distance(points) <= task.near_width)
        out.update(accuracy=float(np.mean(p[uniform] == truth[uniform])),
                   safe=float(np.mean(p[band & (truth == 1)] == 1)), unsafe=float(np.mean(p[band & (truth == 0)] == 0)),
                   near=float(.5*(np.mean(p[near & (truth == 1)] == 1)+np.mean(p[near & (truth == 0)] == 0))))
    else:
        out['stderr'] = record['producer_stderr'][-1500:]
    return out


if __name__ == '__main__':
    counts = {'unicycle_paper_constraints': int(sys.argv[1]) if len(sys.argv) > 1 else 6,
              'quadcopter_hourglass': int(sys.argv[2]) if len(sys.argv) > 2 else 4}
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else ['reference', 'random']
    results = []
    # One method at a time: concurrent runs never share an instance's oracle cache.
    for name in names:
        jobs = [(name, exp, i) for exp, n in counts.items() for i in range(n)]
        with ProcessPoolExecutor(max_workers=8) as pool:
            for out in pool.map(one, jobs):
                print(json.dumps(out), flush=True); results.append(out)
    for exp in counts:
        for name in names:
            rows = [r for r in results if r['experiment'] == exp and r['method'] == name and not r['failure']]
            if rows:
                print(exp, name, 'n=%d acc %.4f safe %.3f unsafe %.3f near %.3f demos %.1f sec %.0f' % (
                    len(rows), *[np.mean([r[k] for r in rows]) for k in ('accuracy', 'safe', 'unsafe', 'near', 'demos', 'seconds')]), flush=True)
    Path('tmp/smoke_systems.json').write_text(json.dumps(results, indent=1))
