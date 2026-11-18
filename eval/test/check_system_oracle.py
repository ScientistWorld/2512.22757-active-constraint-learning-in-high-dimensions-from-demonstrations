"""Job-only checks of the unicycle/quadrotor expert: feasibility, P1 stationarity
in position space, and exact consistency of the lifted states and controls
with independently written forward models. No method code is imported."""
import numpy as np
from system_tasks import SystemTask, EXPERIMENTS, SETTINGS, QUAD_MASS, QUAD_INERTIA, GRAVITY, DT


def unicycle_step(x, u):
    return np.array([x[0]+DT*x[3]*np.cos(x[2]), x[1]+DT*x[3]*np.sin(x[2]), x[2]+DT*u[0], x[3]+DT*u[1]])


def quadrotor_step(x, u):
    p, v, (phi, theta, psi), w = x[:3], x[3:6], x[6:9], x[9:]
    rz = np.array([np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi),
                   np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi), np.cos(theta)*np.cos(phi)])
    W = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)], [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
    dw = (u[1:]-np.cross(w, QUAD_INERTIA*w))/QUAD_INERTIA
    return np.r_[p+DT*v, v+DT*(rz*u[0]/QUAD_MASS-[0, 0, GRAVITY]), x[6:9]+DT*W@w, w+DT*dw]


summary = {}
for experiment in EXPERIMENTS:
    s = SETTINGS[experiment]; worst = {'dynamics': 0., 'stationarity': 0., 'violation': 0.}
    tight = 0
    for index in range(len(s['families'])):
        task = SystemTask(experiment, index)
        step = unicycle_step if task.system == 'unicycle' else quadrotor_step
        for demo in task.initial_demonstrations():
            p = np.asarray(demo['positions']); x = np.asarray(demo['states']); u = np.asarray(demo['controls'])
            assert p.shape == (s['horizon'], s['dim']) and x.shape == (s['horizon'], s['state_dim'])
            assert u.shape == (s['horizon']-1, s['control_dim']) and np.allclose(x[:, :s['dim']], p)
            worst['dynamics'] = max(worst['dynamics'], max(float(np.max(np.abs(step(x[t], u[t])-x[t+1]))) for t in range(len(u))))
            values, gradients = task.value_gradient(p[1:-1])
            worst['violation'] = max(worst['violation'], float(np.max(values)))
            objective = 2*(2*p[1:-1]-p[:-2]-p[2:])
            multipliers = np.maximum(0, -np.sum(objective*gradients, axis=1)/np.sum(gradients**2, axis=1))
            worst['stationarity'] = max(worst['stationarity'], float(np.max(np.abs(objective+multipliers[:, None]*gradients))))
            tight += int(np.sum(np.abs(objective).sum(axis=1) > 1e-4))
        points, truth, uniform = task.evaluation()
        assert points.shape == (s['n_uniform']+s['n_band'], s['dim']) and np.all(np.abs(points) <= s['bound'])
        assert 0 < truth[uniform].mean() < 1 and .2 < truth[~uniform].mean() < .8
        near = task.boundary_distance(points[~uniform]) <= task.near_width
        assert .15 < near.mean() < .6, near.mean()
        inside = points[truth == 0][0]
        assert task.demonstrate(inside, np.full(s['dim'], s['bound']-.5)) is None
    assert worst['dynamics'] < 1e-9 and worst['violation'] < 1e-7 and worst['stationarity'] < 1e-3, worst
    assert tight > 0
    summary[experiment] = {**worst, 'tight_states_in_initial_demos': tight}
print('Unicycle/quadrotor expert feasibility, P1 stationarity and lifted dynamics: pass', summary, flush=True)
