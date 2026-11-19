"""Development unicycle and quadrotor tasks on the paper's App. B.1 constraints.

Visible development instances of g1, g2, g3 (4D unicycle, T=30) and of the
hourglass g5 (12D quadrotor, T=20), each rotated, rescaled, translated and
smoothly deformed. Independent seeds and a different deformation family from
the hidden test. The evaluator owns the expert: it solves Prob. 1
(smoothness cost, position start/goal, the unknown constraint at every time
step) and returns the full state/control trajectory of the system. Nothing is
imported from data/, method/, baseline/ or the test evaluator.

With unbounded controls and free non-position endpoint states, the dynamics
place no restriction on the position sequence: every position path lifts to
an exact state/control trajectory (unicycle: heading and speed from each step;
quadrotor: thrust and attitude from each acceleration, rates and torques from
attitude changes). Prob. 1 is therefore solved exactly in position space and
then lifted.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

TASK_SEED = 30551187
EXPERIMENTS = ('unicycle_paper_constraints', 'quadcopter_hourglass')
BUDGET = 12
REQUEST_LIMIT = 40
DT = 1.
GRAVITY = 9.81
QUAD_MASS = 0.468
QUAD_INERTIA = np.array([4.856e-3, 4.856e-3, 8.801e-3])
SETTINGS = {
    # Paper Sec. 4.2: unicycle, g1-g3, T=30 (Sec. 4.1 default), n_s=2500
    # uniform constraint states; here 4096 uniform plus a boundary band.
    'unicycle_paper_constraints': dict(
        system='unicycle', dim=2, state_dim=4, control_dim=2, horizon=30, bound=15.,
        center_range=2., families=('g1', 'g2', 'g3'), tasks=24, n_uniform=4096,
        n_band=8192, band=.5, near=.175, grid=.25, margin=.3,
        crossings=(([-13.5, -11.5], [13.5, 11.5]), ([-13.5, 11.5], [13.5, -11.5]),
                   ([-14., 1.5], [14., -1.5]), ([1.5, 14.], [-1.5, -14.]))),
    # Paper Sec. 4.2: quadrotor, hourglass g5, T=20, n_s=27,000.
    'quadcopter_hourglass': dict(
        system='quadcopter', dim=3, state_dim=12, control_dim=4, horizon=20, bound=6.,
        center_range=.75, families=('g5',), tasks=24, n_uniform=8192,
        n_band=8192, band=.35, near=.125, grid=.3, margin=.25,
        crossings=(([-5.5, -5.5, -5.5], [5.5, 5.5, 5.5]), ([-5.5, 5.5, -5.5], [5.5, -5.5, 5.5]),
                   ([5.5, -5.5, -5.5], [-5.5, 5.5, 5.5]), ([5.5, 5.5, -5.5], [-5.5, -5.5, 5.5]))),
}
# Shape-frame half extents of the base constraints (for band sampling).
EXTENT = {'g1': 8.3, 'g2': 9.1, 'g3': 7.3, 'g5': 3.7}
# Development deformation: a smooth sinusoidal displacement field in the shape
# frame, z -> z + sum_j a_j sin(omega_j . z + phase_j). The hidden test uses a
# different private field of comparable size.
# RMS displacement about 0.32 (2D) and 0.52 (3D) position units on the
# boundary, matched to the hidden test's field.
WAVES = 3
WAVE_AMPLITUDE = {2: .3, 3: .47}
WAVE_LENGTH = {2: 12., 3: 11.}
Q2 = np.array([[.0868, .0243], [.0243, .0868]])
QQ2 = np.array([-.8403, -.7153])
R2 = 1.7534


def base_constraint(family, z):
    """Paper App. B.1 constraints (unsafe where g>0), value and gradient."""
    if family == 'g1':
        x, y = z.T
        u1 = 1.732*x+y; u2 = x-1.732*y
        value = -.02*np.abs(u1)**1.4-.042*np.abs(u2)**1.4+1
        d1 = -.02*1.4*np.abs(u1)**.4*np.sign(u1); d2 = -.042*1.4*np.abs(u2)**.4*np.sign(u2)
        return value, np.column_stack([1.732*d1+d2, d1-1.732*d2])
    if family == 'g2':
        quad = np.einsum('ni,ij,nj->n', z, Q2, z)
        f1 = quad+z@QQ2+R2; f2 = quad-z@QQ2+R2
        first = f1 <= f2
        value = -np.where(first, f1, f2)
        grad = -(2*z@Q2+np.where(first[:, None], QQ2, -QQ2))
        return value, grad
    if family == 'g3':
        x, y = z.T
        a = .25*x*x+y*y-25
        return (-a*a-50*x*x+687.5)/687.5, np.column_stack([-a*x-100*x, -4*a*y])/687.5
    if family == 'g5':
        x, y, w = z.T
        a = (x*x+y*y)/9+w*w-6
        value = (-a*a-16/3*(x*x+y*y)+39.6)/39.6
        grad = np.column_stack([-4*a*x/9-32/3*x, -4*a*y/9-32/3*y, -4*a*w])/39.6
        return value, grad
    raise ValueError(family)


def random_rotation(rng, dim):
    if dim == 2:
        angle = rng.uniform(-np.pi, np.pi)
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class SystemTask:
    def __init__(self, experiment, index):
        s = SETTINGS[experiment]
        self.experiment = experiment; self.index = index; self.settings = s
        self.dim = d = s['dim']; self.horizon = s['horizon']; self.bound = s['bound']
        self.system = s['system']; self.band_width = s['band']; self.near_width = s['near']
        self.family = s['families'][index % len(s['families'])]
        rng = np.random.default_rng(np.random.SeedSequence([TASK_SEED, 10+EXPERIMENTS.index(experiment), index]))
        self.rotation = random_rotation(rng, d)
        self.scale = rng.uniform(.88, 1.08)
        self.center = rng.uniform(-s['center_range'], s['center_range'], d)
        extra = np.random.default_rng(np.random.SeedSequence([TASK_SEED, 10+EXPERIMENTS.index(experiment), index, 1]))
        direction = extra.normal(size=(WAVES, d))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.wave_vector = direction*2*np.pi/WAVE_LENGTH[d]
        self.wave_phase = extra.uniform(-np.pi, np.pi, WAVES)
        self.wave_amplitude = extra.uniform(-WAVE_AMPLITUDE[d], WAVE_AMPLITUDE[d], (WAVES, d))
        # Evaluation coordinates come from their own stream: evaluation() is
        # idempotent however often (and in whatever order) it is called.
        self.evaluation_seed = np.random.SeedSequence([TASK_SEED, 10+EXPERIMENTS.index(experiment), index, 2])
        self._graph = None; self._evaluation = None

    # ----- the hidden constraint -----
    def value_gradient(self, points):
        q = np.atleast_2d(np.asarray(points, float))
        z = (q-self.center)@self.rotation/self.scale
        angle = z@self.wave_vector.T+self.wave_phase
        warped = z+np.sin(angle)@self.wave_amplitude
        # Jacobian of the warp: I + sum_j a_j (x) cos(angle_j) omega_j.
        jac = np.eye(self.dim)[None]+np.einsum('ji,nj,jk->nik', self.wave_amplitude, np.cos(angle), self.wave_vector)
        value, grad0 = base_constraint(self.family, warped)
        grad_z = np.einsum('ni,nij->nj', grad0, jac)
        return value, grad_z@self.rotation.T/self.scale

    def displacement(self, points):
        """Size of the development deformation at these points (position units)."""
        z = (np.atleast_2d(points)-self.center)@self.rotation/self.scale
        return np.linalg.norm(np.sin(z@self.wave_vector.T+self.wave_phase)@self.wave_amplitude, axis=1)*self.scale

    def project(self, points, steps=60):
        """Damped Newton projection onto g=0; returns points and convergence."""
        q = np.array(points, float)
        for _ in range(steps):
            value, grad = self.value_gradient(q)
            norm2 = np.maximum(np.sum(grad**2, axis=1), 1e-300)
            move = -(value/norm2)[:, None]*grad
            length = np.linalg.norm(move, axis=1)
            move *= np.minimum(1., 1./np.maximum(length, 1e-300))[:, None]
            q = q+move
        value, grad = self.value_gradient(q)
        ok = np.isfinite(value) & (np.abs(value) <= 1e-7*np.linalg.norm(grad, axis=1))
        return q, ok

    def boundary_distance(self, points):
        projected, ok = self.project(points)
        return np.where(ok, np.linalg.norm(np.atleast_2d(points)-projected, axis=1), np.inf)

    def band_points(self, count, rng):
        s = self.settings
        half = (EXTENT[self.family]+1.)*self.scale
        found = []
        total = 0
        while total < count:
            candidates = self.center+rng.uniform(-half, half, (2*count, self.dim))
            projected, ok = self.project(candidates)
            ok &= np.all(np.abs(projected) <= self.bound-s['band'], axis=1)
            found.append(projected[ok]); total += int(ok.sum())
        boundary = np.vstack(found)[:count]
        normal = self.value_gradient(boundary)[1]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        offset = rng.uniform(-s['band'], s['band'], count)
        return boundary+offset[:, None]*normal

    def evaluation(self):
        if self._evaluation is not None:
            return tuple(a.copy() for a in self._evaluation)
        s = self.settings; rng = np.random.default_rng(self.evaluation_seed)
        uniform = rng.uniform(-self.bound, self.bound, (s['n_uniform'], self.dim))
        band = self.band_points(s['n_band'], rng)
        points = np.vstack([uniform, band]); is_uniform = np.arange(len(points)) < s['n_uniform']
        order = rng.permutation(len(points))
        points = points[order]
        truth = (self.value_gradient(points)[0] <= 0).astype(int)
        self._evaluation = (points, truth, is_uniform[order])
        return tuple(a.copy() for a in self._evaluation)

    # ----- the protocol header -----
    def header(self):
        s = self.settings
        model = {'dt': DT}
        if self.system == 'quadcopter':
            model.update(mass=QUAD_MASS, inertia=QUAD_INERTIA.tolist(), gravity=GRAVITY)
        return {'system': self.system, 'dimension': self.dim, 'state_dimension': s['state_dim'],
                'control_dimension': s['control_dim'], 'model': model,
                'bounds': [[-self.bound, self.bound]]*self.dim, 'horizon': self.horizon, 'dt': DT}

    # ----- the expert demonstrator (Prob. 1) -----
    def graph(self):
        if self._graph is None:
            s = self.settings; h = s['grid']; d = self.dim
            axis = np.arange(-self.bound, self.bound+h/2, h)
            shape = (len(axis),)*d
            nodes = np.column_stack([m.ravel() for m in np.meshgrid(*([axis]*d), indexing='ij')])
            value, grad = self.value_gradient(nodes)
            free = value+s['margin']*np.linalg.norm(grad, axis=1) <= 0
            index = np.arange(len(nodes)).reshape(shape)
            rows = []; cols = []; weights = []
            for offset in np.ndindex(*([3]*d)):
                step = np.array(offset)-1
                if not np.any(step): continue
                src = tuple(slice(max(0, -k), len(axis)-max(0, k)) for k in step)
                dst = tuple(slice(max(0, k), len(axis)-max(0, -k)) for k in step)
                a = index[src].ravel(); b = index[dst].ravel()
                keep = free[a] & free[b]
                rows.append(a[keep]); cols.append(b[keep])
                weights.append(np.full(int(keep.sum()), h*np.linalg.norm(step)))
            graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(len(nodes), len(nodes))).tocsr()
            free_index = np.flatnonzero(free)
            self._graph = (graph, nodes, free_index, cKDTree(nodes[free_index]))
        return self._graph

    def route(self, start, goal):
        """Shortest grid path around the margin-inflated obstacle (warm start)."""
        graph, nodes, free_index, tree = self.graph()
        source = free_index[tree.query(start)[1]]; target = free_index[tree.query(goal)[1]]
        distance, predecessor = dijkstra(graph, indices=source, return_predecessors=True)
        if not np.isfinite(distance[target]): return None
        path = [target]
        while path[-1] != source:
            path.append(predecessor[path[-1]])
        polyline = np.vstack([start, nodes[path[::-1]], goal])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
        samples = np.linspace(0, arc[-1], self.horizon)
        return np.column_stack([np.interp(samples, arc, polyline[:, k]) for k in range(self.dim)])

    def demonstrate(self, start, goal):
        start = np.asarray(start, float); goal = np.asarray(goal, float)
        if np.any(self.value_gradient(np.vstack([start, goal]))[0] > 1e-8) or np.linalg.norm(start-goal) < .15:
            return None
        T = self.horizon; d = self.dim
        line = np.linspace(start, goal, T)
        def unpack(z): return np.vstack([start, z.reshape(-1, d), goal])
        def cost(z): return np.sum(np.diff(unpack(z), axis=0)**2)
        def jac(z):
            p = unpack(z); return (2*(2*p[1:-1]-p[:-2]-p[2:])).ravel()
        def constraint(z): return -self.value_gradient(unpack(z)[1:-1])[0]
        def constraint_jac(z):
            grad = -self.value_gradient(unpack(z)[1:-1])[1]
            mat = np.zeros((T-2, d*(T-2)))
            k = np.arange(T-2)
            for j in range(d): mat[k, d*k+j] = grad[:, j]
            return mat
        positions = None
        if np.min(constraint(line[1:-1].ravel())) >= 0:
            # The feasible straight line is the exact unconstrained minimizer.
            positions = line
        else:
            warm = self.route(start, goal)
            if warm is None: return None
            result = minimize(cost, warm[1:-1].ravel(), jac=jac, method='SLSQP',
                              constraints=[{'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}],
                              options={'maxiter': 500, 'ftol': 2e-12})
            if result.success and np.min(constraint(result.x)) >= -1e-7:
                positions = unpack(result.x)
        if positions is None:
            return None
        return self.lift(positions)

    def lift(self, positions):
        """Exact state/control trajectory of the system through these positions."""
        p = np.asarray(positions, float)
        if self.system == 'unicycle':
            step = np.diff(p, axis=0)
            speed = np.linalg.norm(step, axis=1)
            heading = np.arctan2(step[:, 1], step[:, 0])
            for t in range(len(heading)):
                if speed[t] < 1e-12: heading[t] = heading[t-1] if t else 0.
            heading = np.unwrap(heading)
            theta = np.r_[heading, heading[-1]]; v = np.r_[speed, speed[-1]]
            states = np.column_stack([p, theta, v])
            controls = np.column_stack([np.diff(theta), np.diff(v)])/DT
            return {'positions': p.tolist(), 'states': states.tolist(), 'controls': controls.tolist()}
        velocity = np.diff(p, axis=0)/DT
        velocity = np.vstack([velocity, velocity[-1]])
        acceleration = np.diff(velocity, axis=0)/DT
        thrust_vector = acceleration+np.array([0., 0., GRAVITY])
        thrust = QUAD_MASS*np.linalg.norm(thrust_vector, axis=1)
        unit = thrust_vector/np.linalg.norm(thrust_vector, axis=1, keepdims=True)
        roll = -np.arcsin(np.clip(unit[:, 1], -1, 1)); pitch = np.arctan2(unit[:, 0], unit[:, 2])
        euler = np.column_stack([roll, pitch, np.zeros_like(roll)])
        euler = np.vstack([euler, euler[-1]])
        rates = []
        for t in range(len(p)-1):
            phi, theta = euler[t, 0], euler[t, 1]
            w = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                          [0, np.cos(phi), -np.sin(phi)],
                          [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
            rates.append(np.linalg.solve(w*DT, euler[t+1]-euler[t]))
        rates.append(rates[-1])
        rates = np.asarray(rates)
        jw = QUAD_INERTIA*rates
        torque = QUAD_INERTIA*np.diff(rates, axis=0)/DT+np.cross(rates[:-1], jw[:-1])
        states = np.hstack([p, velocity, euler, rates])
        controls = np.column_stack([thrust, torque])
        return {'positions': p.tolist(), 'states': states.tolist(), 'controls': controls.tolist()}

    def initial_demonstrations(self):
        result = [self.demonstrate(np.asarray(s, float), np.asarray(g, float))
                  for s, g in self.settings['crossings']]
        if any(r is None for r in result):
            raise RuntimeError('Fixed initial demonstration failed')
        return result
