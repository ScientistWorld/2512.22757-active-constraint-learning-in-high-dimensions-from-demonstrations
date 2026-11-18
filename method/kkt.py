"""Known dynamics and the KKT Problems 7-8 (App. A.1) over the full trajectory.

A demonstration xi=(x_1..x_T, u_1..u_{T-1}) solves Prob. 1 with the smoothness
cost c=sum_t ||p_{t+1}-p_t||^2, known dynamics h_t: x_{t+1}-f(x_t,u_t)=0 and
position start/goal constraints. The unknown constraint g(phi_sep(x_t)) only
depends on the positions phi_sep(x_t), so every stationarity row of a control
or of a non-position state coordinate is free of it. Those rows must hold with
the dynamics multipliers alone; the remaining (position) rows carry the
unknown term lambda_t * grad g.

Problem 7 (tightness at t): min ||s_{p_t}(xi, 0, nu)||_1 over the dynamics and
endpoint multipliers nu satisfying all unknown-free rows. A strictly positive
optimum (above the numerical tolerance) certifies a tight state.
Problem 8 (gradient estimate): with lambda=1 at the tight times, find w_t with
s_{p_t}+w_t=0 at tight t and s_{p_t}=0 elsewhere (least L1 violation, which is
zero up to the tolerance), i.e. w_t is the KKT-compatible constraint gradient
up to a positive scale.

Both are linear programs in nu. They are solved exactly in the affine space of
the unknown-free rows (a null-space parameterization); an L1 linear program is
solved whenever that space still moves a position row.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

GRAVITY = 9.81
# Quadrotor constants of Sabatino (2015): mass and diagonal inertia.
QUAD_MASS = 0.468
QUAD_INERTIA = np.array([4.856e-3, 4.856e-3, 8.801e-3])


class DoubleIntegrator:
    """x=(p,v), u=a: p+=p+v dt, v+=v+a dt."""
    def __init__(self, dim, dt=1.):
        self.nc = dim; self.n = 2*dim; self.m = dim; self.dt = dt
        self.position = np.arange(dim)

    def trajectory(self, demo):
        p = np.asarray(demo['positions'], float)
        if 'states' in demo:
            return np.asarray(demo['states'], float), np.asarray(demo['controls'], float)
        v = np.asarray(demo['velocities'], float)
        return np.hstack([p, v]), np.asarray(demo['controls'], float)

    def jacobians(self, x, u):
        d = self.nc; eye = np.eye(d); zero = np.zeros((d, d))
        a = np.block([[eye, self.dt*eye], [zero, eye]])
        b = np.vstack([zero, self.dt*eye])
        return a, b


class Unicycle:
    """4D unicycle x=(px,py,theta,v), u=(omega,a)."""
    nc = 2; n = 4; m = 2

    def __init__(self, dt=1.):
        self.dt = dt; self.position = np.arange(2)

    def trajectory(self, demo):
        return np.asarray(demo['states'], float), np.asarray(demo['controls'], float)

    def jacobians(self, x, u):
        dt = self.dt; theta, v = x[2], x[3]
        a = np.eye(4)
        a[0, 2] = -dt*v*np.sin(theta); a[0, 3] = dt*np.cos(theta)
        a[1, 2] = dt*v*np.cos(theta); a[1, 3] = dt*np.sin(theta)
        b = np.zeros((4, 2)); b[2, 0] = dt; b[3, 1] = dt
        return a, b


class Quadcopter:
    """12D quadrotor x=(p, v, eta, omega), u=(thrust, torque).

    p world position, v world velocity, eta=(roll, pitch, yaw) ZYX Euler
    angles, omega body rates. Euler discretization with step dt:
      p+ = p + v dt
      v+ = v + (R(eta) e3 F/m - g e3) dt
      eta+ = eta + W(eta) omega dt
      omega+ = omega + J^{-1}(tau - omega x J omega) dt
    """
    nc = 3; n = 12; m = 4

    def __init__(self, dt=1.):
        self.dt = dt; self.position = np.arange(3)

    def trajectory(self, demo):
        return np.asarray(demo['states'], float), np.asarray(demo['controls'], float)

    def step(self, x, u):
        dt = self.dt
        p, v, eta, w = x[0:3], x[3:6], x[6:9], x[9:12]
        phi, theta, psi = eta[0], eta[1], eta[2]
        thrust, tau = u[0], u[1:4]
        z_axis = np.array([np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi),
                           np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi),
                           np.cos(theta)*np.cos(phi)])
        acc = z_axis*thrust/QUAD_MASS-np.array([0., 0., GRAVITY])
        rates = np.array([w[0]+np.sin(phi)*np.tan(theta)*w[1]+np.cos(phi)*np.tan(theta)*w[2],
                          np.cos(phi)*w[1]-np.sin(phi)*w[2],
                          (np.sin(phi)*w[1]+np.cos(phi)*w[2])/np.cos(theta)])
        jw = QUAD_INERTIA*w
        cross = np.array([w[1]*jw[2]-w[2]*jw[1], w[2]*jw[0]-w[0]*jw[2], w[0]*jw[1]-w[1]*jw[0]])
        dw = (tau-cross)/QUAD_INERTIA
        return np.concatenate([p+v*dt, v+acc*dt, eta+rates*dt, w+dw*dt])

    def jacobians(self, x, u):
        # Complex-step differentiation: exact to machine precision for this
        # analytic map (no subtraction error).
        h = 1e-30
        a = np.empty((12, 12)); b = np.empty((12, 4))
        for j in range(12):
            e = np.zeros(12, complex); e[j] = 1j*h
            a[:, j] = self.step(x+e, u.astype(complex)).imag/h
        for j in range(4):
            e = np.zeros(4, complex); e[j] = 1j*h
            b[:, j] = self.step(x.astype(complex), u+e).imag/h
        return a, b


def make_dynamics(system, dim, dt=1.):
    if system == 'double_integrator':
        return DoubleIntegrator(dim, dt)
    if system == 'unicycle':
        return Unicycle(dt)
    if system == 'quadcopter':
        return Quadcopter(dt)
    raise ValueError('unknown system '+str(system))


def _l1_fit(c, m):
    """min_z ||c + m z||_1 as a linear program; returns (value, z)."""
    rows, k = m.shape
    cost = np.r_[np.zeros(k), np.ones(rows)]
    eye = np.eye(rows)
    a_ub = np.block([[m, -eye], [-m, -eye]])
    b_ub = np.r_[-c, c]
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)]*k+[(0, None)]*rows, method='highs')
    if not res.success:
        raise RuntimeError('KKT linear program failed')
    return float(res.fun), res.x[:k]


def kkt_tight_points(demo, dynamics, tolerance=1e-4):
    """Problems 7 and 8 for one demonstration.

    Returns the tight constraint states (positions) and the P8 gradient
    estimates w_t restricted to the constraint-state coordinates.
    """
    x, u = dynamics.trajectory(demo)
    T, n = x.shape; m = dynamics.m; nc = dynamics.nc; pos = dynamics.position
    p = x[:, pos]
    nx = T*n; nxi = nx+(T-1)*m
    # Smoothness-cost gradient (only position coordinates).
    grad_c = np.zeros(nxi)
    diff = np.diff(p, axis=0)
    gp = np.zeros_like(p); gp[:-1] -= 2*diff; gp[1:] += 2*diff
    for t in range(T):
        grad_c[t*n+pos] = gp[t]
    # J_h^T: dynamics multipliers nu_t (n each), endpoint multipliers (nc each).
    nnu = (T-1)*n+2*nc
    jt = np.zeros((nxi, nnu))
    for t in range(T-1):
        a, b = dynamics.jacobians(x[t], u[t])
        col = slice(t*n, (t+1)*n)
        jt[(t+1)*n:(t+2)*n, col] += np.eye(n)
        jt[t*n:(t+1)*n, col] -= a.T
        jt[nx+t*m:nx+(t+1)*m, col] -= b.T
    jt[np.ix_(pos, (T-1)*n+np.arange(nc))] = np.eye(nc)
    jt[np.ix_((T-1)*n+pos, (T-1)*n+nc+np.arange(nc))] = np.eye(nc)
    position_rows = np.zeros(nxi, bool)
    for t in range(T):
        position_rows[t*n+pos] = True
    free = ~position_rows
    # Affine space of multipliers satisfying every unknown-free row.
    a_free = jt[free]; c_free = grad_c[free]
    nu0 = np.linalg.lstsq(a_free, -c_free, rcond=None)[0]
    if np.max(np.abs(a_free@nu0+c_free), initial=0.) > 1e-8*(1+np.max(np.abs(c_free), initial=0.)):
        raise RuntimeError('demonstration violates the unknown-free KKT rows')
    basis = null_space(a_free)
    scale = 1+np.max(np.abs(jt))
    rows = [t*n+pos for t in range(T)]
    offset = [grad_c[r]+jt[r]@nu0 for r in rows]
    moving = [jt[r]@basis for r in rows]
    def residual_value(t):
        if basis.shape[1] == 0 or np.max(np.abs(moving[t])) <= 1e-10*scale:
            return float(np.sum(np.abs(offset[t])))
        return _l1_fit(offset[t], moving[t])[0]
    tight = [t for t in range(1, T-1) if residual_value(t) > tolerance]
    if not tight:
        return np.zeros((0, nc)), np.zeros((0, nc))
    # Problem 8: w_t absorbs the tight rows; the other rows are matched in L1.
    others = [t for t in range(1, T-1) if t not in tight]
    z = np.zeros(basis.shape[1])
    if basis.shape[1] and others:
        c = np.concatenate([offset[t] for t in others]); mm = np.vstack([moving[t] for t in others])
        if np.max(np.abs(mm)) > 1e-10*scale:
            z = _l1_fit(c, mm)[1]
    points = np.asarray([p[t] for t in tight])
    gradients = np.asarray([-(offset[t]+(moving[t]@z if basis.shape[1] else 0.)) for t in tight])
    return points.reshape(-1, nc), gradients.reshape(-1, nc)
