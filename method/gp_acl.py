"""Algorithm 1 of Qiu et al.: GP-ACL, specialized to free double integrators.

Position elimination is exact: v[t]=p[t+1]-p[t], u[t]=v[t+1]-v[t].
With no control/velocity bounds or endpoint velocity constraints (the paper's
position start/goal setting), every position sequence lifts to a feasible
state/control trajectory. Problems 7/8 below are their reduced KKT programs,
not quantile contact detection. No simulator or truth module is imported.
"""
from __future__ import annotations
import argparse
import json
import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, linprog
from scipy.special import ndtr


class GPModel:
    def __init__(self, points, gradients, params=(.6, .1, 1e-5), values_only=False):
        self.points = np.asarray(points).reshape(-1, 2)
        self.gradients = np.asarray(gradients).reshape(-1, 2)
        self.ls, self.signal, self.noise = params
        self.types = np.tile(np.arange(3), len(self.points)) if not values_only else np.zeros(len(self.points), int)
        self.x = np.repeat(self.points, 3, axis=0) if not values_only else self.points
        self.y = np.column_stack([np.zeros(len(self.points)), self.gradients]).ravel() if not values_only else np.zeros(len(self.points))
        diff = self.x[:, None] - self.x[None, :]
        k = self.signal**2 * np.exp(-np.sum(diff**2, axis=2)/(2*self.ls**2))
        ti, tj = self.types[:, None], self.types[None, :]
        for i in (1, 2):
            k = np.where((ti==i)&(tj==0), k * (-diff[:,:,i-1]/self.ls**2), k)
            k = np.where((ti==0)&(tj==i), k * (diff[:,:,i-1]/self.ls**2), k)
        for i in (1, 2):
            for j in (1, 2):
                k = np.where((ti==i)&(tj==j), k * ((i==j)/self.ls**2-diff[:,:,i-1]*diff[:,:,j-1]/self.ls**4), k)
        k.flat[::len(k)+1] += self.noise
        self.chol = cho_factor(k, lower=True, check_finite=False)
        self.alpha = cho_solve(self.chol, self.y, check_finite=False)

    def cross(self, query):
        q = np.atleast_2d(query)
        diff = self.x[:, None] - q[None, :]
        k = self.signal**2*np.exp(-np.sum(diff**2, axis=2)/(2*self.ls**2))
        for d in (1, 2):
            k = np.where(self.types[:,None]==d, k*(-diff[:,:,d-1]/self.ls**2), k)
        return k

    def mean_var(self, query):
        k = self.cross(query)
        m = k.T @ self.alpha
        v = np.maximum(self.signal**2-np.sum(k*cho_solve(self.chol,k,check_finite=False), axis=0), 1e-12)
        return m, v

    def sample_function(self, rng, n_features=1000, alpha=.3):
        omega = rng.normal(0, 1/self.ls, (n_features, 2))
        phase = rng.uniform(0, 2*np.pi, n_features)
        weights = rng.normal(size=n_features)
        scale = self.signal*np.sqrt(2/n_features)
        angles = self.x @ omega.T + phase
        features = scale*np.cos(angles)
        for d in (1, 2):
            rows = self.types==d
            features[rows] = -scale*np.sin(angles[rows])*omega[:,d-1]
        correction = cho_solve(self.chol, features@weights, check_finite=False)
        def sample(q):
            q = np.atleast_2d(q)
            k = self.cross(q)
            return k.T@self.alpha + alpha*(scale*np.cos(q@omega.T+phase)@weights-k.T@correction)
        return sample


def kkt_tight_points(demo, tolerance=2e-5):
    """Appendix A.1, Problems 7 and 8 in exactly eliminated coordinates.

    All free control stationarity equations force the dynamics multipliers to
    zero on interior positions; endpoint multipliers have no interior column.
    Thus P7 is min ||grad_p c||_1, implemented with its LP epigraph. P8 fixes
    lambda_unknown=1 at tight times and finds w satisfying grad_p c+w=0.
    The other positions must have residual below the numerical tolerance.
    """
    p = np.asarray(demo['positions'], float)
    grad = 2*(2*p[1:-1]-p[:-2]-p[2:])
    points, gradients = [], []
    for point, gc in zip(p[1:-1], grad):
        p7 = linprog(np.ones(2), A_ub=-np.eye(2).repeat(2, axis=0),
                     b_ub=np.array([-gc[0],gc[0],-gc[1],gc[1]]),
                     bounds=[(0,None)]*2, method='highs')
        if not p7.success:
            raise RuntimeError('P7 failed')
        if p7.fun > tolerance:
            p8 = linprog(np.zeros(2), A_eq=np.eye(2), b_eq=-gc,
                         bounds=[(None,None)]*2, method='highs')
            if not p8.success:
                raise RuntimeError('P8 failed')
            points.append(point)
            gradients.append(p8.x)
    return np.asarray(points).reshape(-1,2), np.asarray(gradients).reshape(-1,2)


def train_gp(points, gradients, values_only=False):
    if not len(points):
        return None
    def objective(logp):
        try:
            gp = GPModel(points, gradients, np.exp(logp), values_only)
            return .5*gp.y@gp.alpha + np.log(np.diag(gp.chol[0])).sum() + .5*len(gp.y)*np.log(2*np.pi)
        except np.linalg.LinAlgError:
            return 1e30
    init = np.log([.65,.06,2e-6])
    fit = minimize(objective, init, method='L-BFGS-B',
                   bounds=np.log([[.12,2.5],[.0001,3],[1e-9,.02]]),
                   options={'maxiter':60,'ftol':1e-8})
    params = np.exp(fit.x) if np.isfinite(fit.fun) else np.exp(init)
    return GPModel(points, gradients, params, values_only)


def finite_gradient(f, x):
    e = np.eye(2)*1e-5
    return np.array([(f(x+v)[0]-f(x-v)[0])/(2e-5) for v in e])


def solve_queries(gp, rng, bounds, n_samples=2):
    """Problems 3--6; continuous constrained solves, no clipped fallbacks.

    Only numerical initializations use a candidate grid. A failed constrained
    solve generates no pair. Eq. 7 uses the raw sample gradient. The safety
    CDF is P(g<=0)=Phi(-mean/sd); Eq.10's printed positive sign conflicts with
    the paper's g<=0 convention and its stated safety interpretation.
    """
    lo, hi = np.asarray(bounds).T
    init = rng.uniform(lo,hi,(384,2))
    _, variance = gp.mean_var(init)
    pairs=[]
    for _ in range(n_samples):
        f=gp.sample_function(rng,1000,.3)
        vals=f(init)
        candidates=np.argsort(np.abs(vals))[:64]
        starts=init[candidates[np.argsort(variance[candidates])[-6:]]]
        solutions=[]
        for s in starts:
            res=minimize(lambda q:-gp.mean_var(q)[1][0], s, method='SLSQP', bounds=bounds,
                         constraints=[{'type':'eq','fun':lambda q:f(q)[0]}],
                         options={'maxiter':100,'ftol':1e-10})
            if res.success and abs(f(res.x)[0])<1e-6:
                solutions.append(res)
        if not solutions:
            continue
        mi=min(solutions,key=lambda r:r.fun).x
        raw=finite_gradient(f,mi)
        if np.linalg.norm(raw)<1e-8:
            continue
        center=mi+.1*raw  # Equation 7, eta=.1 (not specified by the paper).
        if np.any(center<lo) or np.any(center>hi) or f(center)[0]<=0:
            continue
        # Problem 4 in 2D is a scalar constrained solve along the hyperplane.
        tangent=np.array([-raw[1],raw[0]])/np.linalg.norm(raw)
        p4=[]
        for sign in (-1,1):
            direction=sign*tangent
            upper=min([(hi[d]-center[d])/direction[d] if direction[d]>0 else
                       (lo[d]-center[d])/direction[d] for d in range(2) if abs(direction[d])>1e-10])
            for frac in (.02,.25,.75):
                r=minimize(lambda t:-t[0]**2,[upper*frac],method='SLSQP',bounds=[(0,upper)],
                           constraints=[{'type':'ineq','fun':lambda t:f(center+t[0]*direction)[0]-1e-9}],
                           options={'maxiter':80,'ftol':1e-10})
                if r.success and f(center+r.x[0]*direction)[0]>=-1e-7 and r.x[0]>1e-6:
                    p4.append((r.x[0],r.x[0]*direction))
        directions=[]
        if p4:
            directions.append((max(p4,key=lambda v:v[0])[1],(.55,.55)))
        directions.append((raw,(.30,.55)))
        # Problems 5 / 6: minimize tau^2 with exact sample and chance constraints.
        for direction,betas in directions:
            pair=[]
            for sign,beta in zip((-1,1),betas):
                vector=sign*direction
                upper=min([(hi[d]-center[d])/vector[d] if vector[d]>0 else
                           (lo[d]-center[d])/vector[d] for d in range(2) if abs(vector[d])>1e-12])
                def constraints(t):
                    q=center+t[0]*vector
                    mean,var=gp.mean_var(q)
                    return np.array([-f(q)[0]-.001,ndtr(-mean[0]/np.sqrt(var[0]))-beta])
                endpoints=[]
                for frac in (.2,.65,.99):
                    r=minimize(lambda t:t[0]**2,[upper*frac], method='SLSQP', bounds=[(1e-7,upper)],
                               constraints=[{'type':'ineq','fun':constraints}],
                               options={'maxiter':120,'ftol':1e-10})
                    if r.success and np.min(constraints(r.x))>=-1e-7:
                        endpoints.append(r)
                if not endpoints:
                    break
                pair.append(center+min(endpoints,key=lambda r:r.fun).x[0]*vector)
            if len(pair)==2:
                pairs.append(pair)
    return pairs


def fit_periodic(points, gradients, query, order=12):
    """Honest headroom variant: periodic contact/normal regression.

    No oracle labels, task parameters or test feedback are used. A smooth
    periodic representation is appropriate for approximately radial obstacles;
    the private stress instances include unresolved high frequency structure.
    """
    theta=np.arctan2(points[:,1],points[:,0]); radii=np.linalg.norm(points,axis=1)
    radial=points/radii[:,None]; tangent=np.column_stack([-radial[:,1],radial[:,0]])
    nr=np.sum(gradients*radial,axis=1)
    norms=np.linalg.norm(gradients,axis=1)
    good=nr/np.maximum(norms,1e-12)<-.08
    harmonics=np.arange(1,order+1)
    def design(t):
        z=np.asarray(t)[:,None]*harmonics
        return np.column_stack([np.ones(len(t)),np.sin(z),np.cos(z)])
    z=theta[good,None]*harmonics
    deriv=np.column_stack([np.zeros(good.sum()),np.cos(z)*harmonics,-np.sin(z)*harmonics])
    dr=-radii[good]*np.sum(gradients[good]*tangent[good],axis=1)/nr[good]
    a=np.vstack([design(theta),.25*deriv]); b=np.r_[radii,.25*dr]
    ridge=2e-5*np.r_[.001,harmonics**4,harmonics**4]
    coef=np.linalg.solve(a.T@a+np.diag(ridge),a.T@b)
    radius=design(np.arctan2(query[:,1],query[:,0]))@coef
    return (np.linalg.norm(query,axis=1)>=radius).astype(int)


def send(value):
    print(json.dumps(value,allow_nan=False),flush=True)
    return json.loads(sys.stdin.readline())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='reference', choices=['reference','random','periodic','periodic6','normalized','no_gradients','no_acquisition'])
    args=parser.parse_args()
    task=json.loads(sys.stdin.readline())
    rng=np.random.default_rng(task['seed'])
    demos=list(task['initial_demos'])
    bounds=task['bounds']
    available_requests=task['query_limit']
    for iteration in range(3):
        contacts=[kkt_tight_points(d) for d in demos]
        points=np.vstack([p for p,g in contacts]); gradients=np.vstack([g for p,g in contacts])
        gp=train_gp(points,gradients)
        if iteration==2 or args.variant=='no_acquisition':
            break
        if available_requests<=0:
            continue
        accepted=0
        for query_round in range(6):
            if accepted>=4:
                break
            if args.variant=='random' or gp is None:
                lo,hi=np.asarray(bounds).T
                pairs=[rng.uniform(lo,hi,(2,2)) for _ in range(4-accepted)]
            else:
                pairs=solve_queries(gp,rng,bounds)
            for pair in pairs:
                if accepted>=4:
                    break
                response=send({'op':'query','start':pair[0].tolist(),'goal':pair[1].tolist()})
                available_requests=response.get('requests_remaining',0)
                if response.get('demo'):
                    demos.append(response['demo']); accepted+=1
                if response.get('requests_remaining',1)<=0:
                    break
            if 'response' in locals() and response.get('requests_remaining',1)<=0:
                break
    response=send({'op':'finish'})
    query=np.asarray(response['points'],float)
    if args.variant in ('periodic','periodic6') and len(points)>4:
        pred=fit_periodic(points,gradients,query,6 if args.variant=='periodic6' else 12)
    elif args.variant=='normalized' and len(points):
        normals=.1*gradients/np.maximum(np.linalg.norm(gradients,axis=1,keepdims=True),1e-12)
        alternative=train_gp(points,normals)
        pred=(alternative.mean_var(query)[0]<=0).astype(int)
    elif args.variant=='no_gradients':
        ablation=train_gp(points,gradients,True)
        pred=(ablation.mean_var(query)[0]<=0).astype(int) if ablation else np.ones(len(query),int)
    else:
        pred=(gp.mean_var(query)[0]<=0).astype(int) if gp else np.ones(len(query),int)
    print(json.dumps({'predictions':pred.tolist()},allow_nan=False),flush=True)


if __name__=='__main__':
    main()
