"""Job-only check that the development and hidden-test deformation fields of
the unicycle/quadrotor tasks have comparable size on the constraint boundary
(RMS and 90th percentile displacement over boundary-band states)."""
import importlib.util
import json
import numpy as np


def load(side):
    spec = importlib.util.spec_from_file_location(f'system_tasks_{side}', f'eval/{side}/system_tasks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


out = {}
for side in ('train', 'test'):
    module = load(side)
    for exp in module.EXPERIMENTS:
        values = []
        for index in range(12):
            task = module.SystemTask(exp, index)
            points, truth, uniform = task.evaluation()
            values.append(task.displacement(points[~uniform]))
        values = np.concatenate(values)
        out[f'{side}.{exp}'] = {'rms': round(float(np.sqrt(np.mean(values**2))), 3),
                                'p90': round(float(np.quantile(values, .9)), 3)}
print('Deformation size on boundary bands:', json.dumps(out), flush=True)
