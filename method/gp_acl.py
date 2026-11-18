"""Algorithm 1 of Qiu et al.: GP-ACL for constraint states in 2D or 3D.

Demonstrations come from the evaluator's expert (Prob. 1) for a known system:
the 2D double integrator, the 4D unicycle or the 12D quadrotor. Problems 7-8
are solved as linear programs over the full KKT system of that system's
dynamics (method/kkt.py). The GP, its posterior samples and Problems 3-6 live
in the constraint space (positions, 2D or 3D). No simulator or truth module is
imported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, brentq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kkt import kkt_tight_points, make_dynamics  # noqa: E402

# Memoized GP training results (hyperparameters only, keyed by a digest of
# this source file and the exact training data). Variants that share the
# reference acquisition and differ only in their final classifier reuse the
# reference's fits instead of re-optimizing; no data are stored.
SOURCE_DIGEST=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
FIT_CACHE=Path(os.environ.get('GP_ACL_FIT_CACHE',Path(__file__).resolve().parents[1]/'tmp'/'gp_fit_cache'))


class GPModel:
    """Joint value/derivative RBF GP with a constant prior mean, in d dims.

    The constant has zero derivative, so gradient observations are unchanged;
    value observations are measured relative to it. This is the constant-mean
    derivative GP (GPyTorch ConstantMeanGrad/RBFKernelGrad) used by the GP
    constraint-learning procedure the paper cites for its training.
    """
    def __init__(self, points, gradients, params=(.6, .1, 1e-5), values_only=False, constant=0.):
        points=np.asarray(points,float)
        self.dim=dim=points.shape[-1]
        self.points = points.reshape(-1, dim)
        self.gradients = np.asarray(gradients,float).reshape(-1, dim)
        self.ls, self.signal, self.noise = params
        self.constant = float(constant)
        self.types = np.tile(np.arange(dim+1), len(self.points)) if not values_only else np.zeros(len(self.points), int)
        self.x = np.repeat(self.points, dim+1, axis=0) if not values_only else self.points
        self.y = np.column_stack([np.zeros(len(self.points)), self.gradients]).ravel() if not values_only else np.zeros(len(self.points))
        self.value_mask = (self.types == 0).astype(float)
        diff = self.x[:, None] - self.x[None, :]
        k = self.signal**2 * np.exp(-np.sum(diff**2, axis=2)/(2*self.ls**2))
        ti, tj = self.types[:, None], self.types[None, :]
        for i in range(1, dim+1):
            k = np.where((ti==i)&(tj==0), k * (-diff[:,:,i-1]/self.ls**2), k)
            k = np.where((ti==0)&(tj==i), k * (diff[:,:,i-1]/self.ls**2), k)
        for i in range(1, dim+1):
            for j in range(1, dim+1):
                k = np.where((ti==i)&(tj==j), k * ((i==j)/self.ls**2-diff[:,:,i-1]*diff[:,:,j-1]/self.ls**4), k)
        self.kernel = k.copy()
        k.flat[::len(k)+1] += self.noise
        self.chol = cho_factor(k, lower=True, check_finite=False)
        self.alpha = cho_solve(self.chol, self.y-self.constant*self.value_mask, check_finite=False)

    def cross(self, query):
        q = np.atleast_2d(query)
        diff = self.x[:, None] - q[None, :]
        k = self.signal**2*np.exp(-np.sum(diff**2, axis=2)/(2*self.ls**2))
        for d in range(1, self.dim+1):
            k = np.where(self.types[:,None]==d, k*(-diff[:,:,d-1]/self.ls**2), k)
        return k

    def mean_var(self, query, chunk=4096):
        q=np.atleast_2d(query)
        if len(q)>chunk:
            parts=[self.mean_var(q[i:i+chunk]) for i in range(0,len(q),chunk)]
            return np.concatenate([a for a,b in parts]),np.concatenate([b for a,b in parts])
        k = self.cross(q)
        m = self.constant + k.T @ self.alpha
        v = np.maximum(self.signal**2-np.sum(k*cho_solve(self.chol,k,check_finite=False), axis=0), 1e-12)
        return m, v

    def mean(self, query, chunk=8192):
        q=np.atleast_2d(query)
        return np.concatenate([self.constant+self.cross(q[i:i+chunk]).T@self.alpha for i in range(0,len(q),chunk)])

    def _dk(self, q):
        diff = self.x[:, None] - q[None, :]
        base = self.signal**2*np.exp(-np.sum(diff**2,axis=2)/(2*self.ls**2))
        dk = base[:,:,None]*diff/self.ls**2
        eye=np.eye(self.dim)
        for d in range(1,self.dim+1):
            rows = self.types==d
            dk[rows] = base[rows,:,None]*(eye[d-1]/self.ls**2
                       -diff[rows,:,d-1,None]*diff[rows]/self.ls**4)
        return dk

    def derivatives(self, query):
        """Analytic mean/variance gradients for the continuous P3 solve."""
        q = np.atleast_2d(query)
        dk = self._dk(q)
        k = self.cross(q)
        return np.einsum('iqd,i->qd',dk,self.alpha), -2*np.einsum(
            'iqd,iq->qd',dk,cho_solve(self.chol,k,check_finite=False))

    def sample_function(self, rng, n_features=1000, alpha=.3):
        """Posterior draw of Eq. 16 (Wilson et al. pathwise RFF update)."""
        omega = rng.normal(0, 1/self.ls, (n_features, self.dim))
        phase = rng.uniform(0, 2*np.pi, n_features)
        weights = rng.normal(size=n_features)
        scale = self.signal*np.sqrt(2/n_features)
        angles = self.x @ omega.T + phase
        features = scale*np.cos(angles)
        for d in range(1, self.dim+1):
            rows = self.types==d
            features[rows] = -scale*np.sin(angles[rows])*omega[:,d-1]
        correction = cho_solve(self.chol, features@weights, check_finite=False)
        coefficient = self.alpha-alpha*correction
        def sample(q, chunk=8192):
            q = np.atleast_2d(q)
            out=[]
            for i in range(0,len(q),chunk):
                block=q[i:i+chunk]
                k = self.cross(block)
                out.append(self.constant + k.T@self.alpha + alpha*(scale*np.cos(block@omega.T+phase)@weights-k.T@correction))
            return np.concatenate(out)
        def derivative(q):
            q=np.atleast_2d(q)
            dk=self._dk(q)
            prior=-scale*(np.sin(q@omega.T+phase)*weights)@omega
            return np.einsum('iqd,i->qd',dk,coefficient)+alpha*prior
        sample.derivative=derivative
        return sample


# Chou et al. (2022), Sec. IV-B, which the paper's Sec. 2.3 says its GP
# training parallels: minimize the negative marginal log likelihood plus the
# demonstration-feasibility hinge mean_n max(mu(x_n)+rho*sd(x_n), 0) over all
# demonstrated states. Neither paper reports rho; RHO (--rho) is a mild safety
# margin. Smaller rho moves the tied prior mean (-rho*signal, see CONFIG)
# toward the paper's zero-mean posterior.
RHO = 0.1
# Lower bound of the fitted observation-noise variance (Eq. 15's sigma^2).
# GPyTorch's inferred-noise floor 1e-4 (chosen on the development split among
# 1e-4, 1e-5 and 1e-6; 1e-10 made the training covariance numerically
# singular, with spurious unsafe regions far from all observations).
NOISE_FLOOR = 0.0001


def _covariance_derivatives(gp):
    """dK/dlog(length) for the joint value/derivative training covariance."""
    diff=gp.x[:,None]-gp.x[None,:]
    r2=np.sum(diff**2,axis=2)/gp.ls**2
    base=gp.signal**2*np.exp(-r2/2)
    dlength=gp.kernel*r2
    ti,tj=gp.types[:,None],gp.types[None,:]
    for i in range(1,gp.dim+1):
        mixed=((ti==i)&(tj==0))|((ti==0)&(tj==i))
        dlength=np.where(mixed,gp.kernel*(r2-2),dlength)
    for i in range(1,gp.dim+1):
        for j in range(1,gp.dim+1):
            derivative=base*((i==j)*(r2-2)/gp.ls**2
                -diff[:,:,i-1]*diff[:,:,j-1]*(r2-4)/gp.ls**4)
            dlength=np.where((ti==i)&(tj==j),derivative,dlength)
    return dlength


def training_objective(z, points, gradients, states=(), values_only=False, rho=None):
    """Loss and exact gradient in z=(log length, log signal, log noise, mean).

    The likelihood is averaged over observations (as GPyTorch's exact MLL is)
    and the hinge over demonstrated states, with equal weights as in Chou et
    al. (2022). rho defaults to the run's CONFIG['rho'].
    """
    rho=CONFIG['rho'] if rho is None else rho
    try:
        gp=GPModel(points,gradients,np.exp(z[:3]),values_only,z[3])
    except np.linalg.LinAlgError:
        return 1e30,np.zeros(4)
    count=len(gp.y)
    centered=gp.y-gp.constant*gp.value_mask
    value=(.5*centered@gp.alpha+np.log(np.diag(gp.chol[0])).sum()
           +.5*count*np.log(2*np.pi))/count
    inverse=cho_solve(gp.chol,np.eye(count),check_finite=False)
    residual=inverse-np.outer(gp.alpha,gp.alpha)
    dlength=_covariance_derivatives(gp)
    gradient=np.array([.5*np.sum(residual*dlength)/count,
                       .5*np.sum(residual*(2*gp.kernel))/count,
                       .5*gp.noise*np.trace(residual)/count,
                       -(gp.value_mask@gp.alpha)/count])
    states=np.asarray(states,float).reshape(-1,gp.dim)
    if len(states):
        kx=gp.cross(states)
        v=cho_solve(gp.chol,kx,check_finite=False)
        mean=gp.constant+kx.T@gp.alpha
        raw=gp.signal**2-np.sum(kx*v,axis=0)
        sd=np.sqrt(np.maximum(raw,1e-12))
        hinge=mean+rho*sd
        value+=np.mean(np.maximum(hinge,0))
        active=hinge>0
        if np.any(active):
            ka,va,sa=kx[:,active],v[:,active],sd[active]
            exact=raw[active]>1e-12
            diff=gp.x[:,None]-states[active][None,:]
            r2=np.sum(diff**2,axis=2)/gp.ls**2
            dka=np.where((gp.types==0)[:,None],ka*r2,ka*(r2-2))
            # Length: general formula. Signal: dK=2(K-noise I), dk=2k and
            # K v=k simplify the quadratic form. Noise: dK=noise I, dk=0.
            dmean=[dka.T@gp.alpha-va.T@(dlength@gp.alpha),
                   2*ka.T@gp.alpha-va.T@(2*gp.kernel@gp.alpha),
                   -gp.noise*(va.T@gp.alpha)]
            quadratic_signal=2*np.sum(va*ka,axis=0)-2*gp.noise*np.sum(va*va,axis=0)
            dvariance=[-2*np.sum(dka*va,axis=0)+np.sum(va*(dlength@va),axis=0),
                       2*gp.signal**2-4*np.sum(ka*va,axis=0)+quadratic_signal,
                       gp.noise*np.sum(va*va,axis=0)]
            for j in range(3):
                gradient[j]+=np.sum(dmean[j]+rho*np.where(exact,dvariance[j],0)/(2*sa))/len(states)
            gradient[3]+=np.sum(1-gp.value_mask@va)/len(states)
    return value,gradient


def marginal_likelihood(logp, points, gradients, values_only=False):
    """Zero-mean likelihood term alone (kept for the numerical checks)."""
    value,gradient=training_objective(np.r_[logp,0.],points,gradients,(),values_only)
    return value,gradient[:3]


# Run configuration (command-line flags). prior='tied' (the reference) fixes
# the constant prior mean at -rho*signal, the hinge boundary for states far
# from data, so unexplored states keep prior safety probability Phi(rho),
# above one half and below the paper's endpoint level beta=0.55 (Eq. 10
# endpoints need evidence of safety). prior='zero' is the paper's own
# posterior (Eqs. 15-16, zero prior mean); it was tested as the reference and
# rejected on the development split (scripts/select_prior.py): uniform
# accuracy fell to 0.73 (0.97 with 'tied'). prior='learned' optimizes the
# constant freely (an earlier package).
# noise_floor: lower bound of the fitted observation-noise variance; hinge:
# include the demonstration-feasibility hinge in training (Chou et al.);
# tight_tol: Problem 7 optimal-value threshold for a tight state.
CONFIG={'prior':'tied','trust':1.0,'draws':4,'rho':RHO,'noise_floor':NOISE_FLOOR,'hinge':True,'tight_tol':1e-4}


def tied_objective(y, points, gradients, states=(), values_only=False):
    """Training loss in (log length, log signal, log noise), mean -rho*signal."""
    rho=CONFIG['rho']
    signal=np.exp(y[1])
    value,gradient=training_objective(np.r_[y,-rho*signal],points,gradients,states,values_only,rho)
    out=gradient[:3].copy();out[1]+=gradient[3]*(-rho*signal)
    return value,out


def zero_objective(y, points, gradients, states=(), values_only=False):
    """Training loss in (log length, log signal, log noise), zero prior mean."""
    value,gradient=training_objective(np.r_[y,0.],points,gradients,states,values_only)
    return value,gradient[:3].copy()


FIXED_MEAN={'tied':(tied_objective,lambda signal:-CONFIG['rho']*signal),
            'zero':(zero_objective,lambda signal:0.)}


def _fit_key(points, gradients, states, values_only, init, prior='learned'):
    digest=hashlib.sha256(SOURCE_DIGEST.encode());digest.update(prior.encode())
    digest.update(f"|rho={float(CONFIG['rho'])!r}|floor={float(CONFIG['noise_floor'])!r}|hinge={bool(CONFIG['hinge'])}|".encode())
    for array in (points,gradients,states):
        digest.update(np.ascontiguousarray(array,float).tobytes()); digest.update(b'|')
    digest.update(b'values' if values_only else b'joint')
    digest.update(b'none' if init is None else np.ascontiguousarray(init,float).tobytes())
    return digest.hexdigest()


def train_gp(points, gradients, states=(), values_only=False, init=None, prior=None):
    """Minimize the training loss from deterministic starts (or warm start).

    Starts are in physical position units. The prior mean is fixed for
    prior='tied' (the reference) and 'zero'; a learned constant ('learned')
    starts on the hinge boundary far from data (-rho * signal). With `init`
    (the previous Alg. 1 iteration's optimum) the optimizer is warm-started
    from it plus one default start.
    """
    if not len(points):
        return None
    prior=prior or CONFIG['prior']
    points=np.asarray(points,float); dim=points.shape[-1]
    points=points.reshape(-1,dim); gradients=np.asarray(gradients,float).reshape(-1,dim)
    states=np.asarray(states,float).reshape(-1,dim)
    if not CONFIG['hinge']:
        states=np.zeros((0,dim))
    bounds=[tuple(b) for b in np.log([[.35,12.5],[1e-5,100],[CONFIG['noise_floor'],1.]])]+[(-1e3,1e3)]
    key=_fit_key(points,gradients,states,values_only,init,prior)
    path=FIT_CACHE/f'{key}.json'
    best=None
    try:
        best=np.asarray(json.loads(path.read_text()),float)
        if best.shape!=(4,) or not np.all(np.isfinite(best)): best=None
    except (OSError,ValueError):
        best=None
    if best is None:
        rms=max(float(np.sqrt(np.mean(gradients**2))),1e-4)
        starts=[]
        if init is not None:
            starts.append(np.clip(np.asarray(init,float),[b[0] for b in bounds],[b[1] for b in bounds]))
        for length in ((3.25,) if init is not None else (1.,3.25,6.)):
            signal=max(rms*length,.001)
            start=np.r_[np.log([length,signal,max((.03*rms)**2,1e-8)]),-CONFIG['rho']*signal]
            starts.append(np.clip(start,[b[0] for b in bounds],[b[1] for b in bounds]))
        fits=[]
        fixed=prior in FIXED_MEAN
        objective=FIXED_MEAN[prior][0] if fixed else training_objective
        for start in starts:
            fit=minimize(objective,start[:3] if fixed else start,
                         args=(points,gradients,states,values_only),
                         jac=True,method='L-BFGS-B',bounds=bounds[:3] if fixed else bounds,
                         options={'maxiter':150,'ftol':1e-10,'gtol':1e-6})
            if np.isfinite(fit.fun) and fit.fun<1e29: fits.append(fit)
        if not fits: raise RuntimeError('GP training optimization failed')
        best=min(fits,key=lambda fit:fit.fun).x
        if fixed: best=np.r_[best,FIXED_MEAN[prior][1](np.exp(best[1]))]
        try:
            FIT_CACHE.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix(f'.{os.getpid()}.tmp')
            temp.write_text(json.dumps(best.tolist())); temp.replace(path)
        except OSError:
            pass
    model=GPModel(points, gradients, np.exp(best[:3]), values_only, best[3])
    model.z=best
    return model


def finite_gradient(f, x):
    x=np.asarray(x,float); e = np.eye(len(x))*1e-5
    return np.array([(f(x+v)[0]-f(x-v)[0])/(2e-5) for v in e])


def scalar_feasible_extreme(constraints, upper, length, nearest=True):
    """Solve the scalar P4/P5/P6 restriction by continuous root refinement.

    Squared distance is monotone on each signed ray. Its optimum is a domain
    endpoint or a constraint root. A mesh brackets roots; brentq refines them,
    and feasible intervals (not mesh predictions) select the extremum.
    """
    if upper<=1e-7: return None
    count=min(2048,max(129,int(np.ceil(upper/(length/20)))+1))
    mesh=np.linspace(0,upper,count)
    values=np.asarray(constraints(mesh))
    if values.ndim==1: values=values[:,None]
    edges=[0.,float(upper)]
    for j in range(values.shape[1]):
        changes=np.flatnonzero(np.signbit(values[:-1,j])!=np.signbit(values[1:,j]))
        for k in changes:
            root=brentq(lambda t:np.atleast_2d(constraints(np.array([t])))[0,j],
                        mesh[k],mesh[k+1],xtol=1e-10)
            edges.append(float(root))
        edges.extend(mesh[values[:,j]==0].tolist())
    edges=np.unique(edges)
    feasible=[]
    for a,b in zip(edges[:-1],edges[1:]):
        if np.all(constraints(np.array([(a+b)/2]))>=0):
            value=a if nearest else b
            # Move an infinitesimal distance into this feasible interval to
            # avoid a rounded root being rejected by the final constraints.
            value=value+(1 if nearest else -1)*min(1e-8,(b-a)/4)
            if value>1e-7 and np.all(constraints(np.array([value]))>=-1e-8):
                feasible.append(value)
    if not feasible: return None
    return min(feasible) if nearest else max(feasible)


def endpoint_constraints(gp, sample, query, beta):
    """P5/P6 endpoint inequalities of Eq. 10, as nonnegative residuals.

    Sec. 3.1 defines the avoid set as g>0 and Eq. 10's prose requires
    endpoints that are "safe with respect to the GP posterior with probability
    at least beta", i.e. P(g<=0)=Phi(-mean/sd)>=beta. The printed Phi(+mean/sd)
    is the probability of being unsafe under the paper's own convention (its
    Appendix B.1 likewise misprints the avoid-set sign); using it places
    endpoints inside the inferred obstacle, where the expert cannot start.
    """
    mean, variance = gp.mean_var(query)
    return np.column_stack([(-sample(query)-.001)/gp.signal,
                            ndtr(-mean/np.sqrt(variance))-beta])


# P3 is solved over the posterior sample's boundary within this many fitted
# lengthscales of the nearest tight state, where that boundary is anchored by
# observations (spurious zero contours of a draw far from all data carry no
# information about where demonstrations can touch the constraint).
TRUST_LENGTHSCALES=1.0  # default of CONFIG['trust']
# Posterior draws per Alg. 1 batch (P, unspecified in the paper; chosen on the
# development split); each yields a P4-5 and a P6 pair.
POSTERIOR_DRAWS=4  # default of CONFIG['draws']
# A deterministic expert returns the same demonstration for (nearly) the same
# start/goal pair, so such repeated requests are skipped.
DUPLICATE_TOLERANCE=.25
MAX_BATCHES=10
MAX_REDRAWS=3
# P3 boundary enumeration: grid cells per box side at most (finer in 2D).
GRID_RESOLUTION={2:160,3:48}
MAX_CROSSINGS=4000
# Problem 4 in 3D: the hyperplane H is a plane; its in-plane directions are
# enumerated at this many angles (in 2D H is a line: two directions).
PLANE_DIRECTIONS=36


def boundary_crossings(f, lo, hi, step, chunk=4096):
    """All zero crossings of f along the edges of a regular grid on [lo,hi]."""
    lo=np.asarray(lo,float); hi=np.asarray(hi,float); dim=len(lo)
    axes=[lo[k]+step*np.arange(int(np.floor((hi[k]-lo[k])/step))+1) for k in range(dim)]
    if any(len(a)<2 for a in axes): return np.zeros((0,dim))
    mesh=np.meshgrid(*axes,indexing='ij')
    grid=np.column_stack([m.ravel() for m in mesh])
    v=np.concatenate([np.asarray(f(grid[i:i+chunk])).ravel() for i in range(0,len(grid),chunk)]).reshape(mesh[0].shape)
    found=[]
    for axis in range(dim):
        first=tuple(slice(None,-1) if k==axis else slice(None) for k in range(dim))
        second=tuple(slice(1,None) if k==axis else slice(None) for k in range(dim))
        a=v[first]; b=v[second]
        mask=np.signbit(a)!=np.signbit(b)
        if not np.any(mask): continue
        den=a[mask]-b[mask]
        frac=np.where(den!=0,a[mask]/np.where(den!=0,den,1.),.5)
        found.append(np.column_stack([mesh[k][first][mask]+(frac*step if k==axis else 0) for k in range(dim)]))
    return np.vstack(found) if found else np.zeros((0,dim))


def nearest_distance(points, anchors, chunk=2048):
    out=np.empty(len(points))
    for i in range(0,len(points),chunk):
        block=points[i:i+chunk]
        out[i:i+chunk]=np.sqrt(np.min(np.sum((block[:,None,:]-anchors[None,:,:])**2,axis=2),axis=1))
    return out


def project_to_boundary(f, q, signal, steps=30):
    """Newton projection of q onto f=0 along the sample gradient."""
    q=np.asarray(q,float).copy()
    for _ in range(steps):
        value=f(q)[0]
        if abs(value)/signal<1e-8: return q
        gradient=f.derivative(q)[0]; norm=gradient@gradient
        if norm<1e-20: return None
        q=q-value*gradient/norm
    return q if abs(f(q)[0])/signal<1e-6 else None


def solve_p3(gp, f, lo, hi):
    """Problem 3: argmax posterior variance on {f=0} near the observations.

    The constrained global maximum is found by enumerating the sampled
    boundary on a grid finer than the fitted lengthscale (zero crossings,
    linear interpolation), then polishing the best distinct candidates with
    the continuous constrained NLP. Starting the local solver from observed
    tight states is avoided: the posterior variance is minimal, with zero
    gradient, at observations, so a local solver started there returns them.
    """
    signal=max(gp.signal,1e-10)
    contacts=gp.points
    reach=CONFIG['trust']*gp.ls
    box_lo=np.maximum(lo,contacts.min(axis=0)-reach); box_hi=np.minimum(hi,contacts.max(axis=0)+reach)
    if np.any(box_hi-box_lo<=0): return None
    step=float(np.clip(gp.ls/4,.05,.5)); step=max(step,float(np.max(box_hi-box_lo))/GRID_RESOLUTION[gp.dim])
    crossings=boundary_crossings(f,box_lo,box_hi,step)
    if not len(crossings): return None
    crossings=crossings[nearest_distance(crossings,contacts)<=reach]
    if not len(crossings): return None
    if len(crossings)>MAX_CROSSINGS:
        # Deterministic thinning (separate stream: the run's draws are unchanged).
        keep=np.random.default_rng(len(crossings)).choice(len(crossings),MAX_CROSSINGS,replace=False)
        crossings=crossings[np.sort(keep)]
    variance=gp.mean_var(crossings)[1]
    chosen=[]
    for i in np.argsort(-variance):
        if all(np.linalg.norm(crossings[i]-crossings[j])>=gp.ls/2 for j in chosen): chosen.append(i)
        if len(chosen)==3: break
    best=None
    for i in chosen:
        start=crossings[i]
        local=list(zip(np.maximum(lo,start-step),np.minimum(hi,start+step)))
        res=minimize(lambda q:-gp.mean_var(q)[1][0]/signal**2,start,
                     jac=lambda q:-gp.derivatives(q)[1][0]/signal**2,
                     method='SLSQP',bounds=local,
                     constraints=[{'type':'eq','fun':lambda q:f(q)[0]/signal,
                                   'jac':lambda q:f.derivative(q)[0]/signal}],
                     options={'maxiter':100,'ftol':1e-10})
        q=res.x if (res.success and abs(f(res.x)[0])/signal<1e-6) else project_to_boundary(f,start,signal)
        if q is None or np.any(q<lo) or np.any(q>hi): continue
        value=gp.mean_var(q)[1][0]
        if best is None or value>best[0]: best=(value,q)
    return None if best is None else best[1]


def hyperplane_directions(normal):
    """Unit directions spanning H (Eq. 8), the hyperplane orthogonal to normal."""
    norm=np.linalg.norm(normal)
    if len(normal)==2:
        tangent=np.array([-normal[1],normal[0]])/norm
        return [-tangent,tangent]
    unit=normal/norm
    helper=np.eye(len(normal))[int(np.argmin(np.abs(unit)))]
    first=np.cross(unit,helper); first/=np.linalg.norm(first)
    second=np.cross(unit,first)
    angles=np.linspace(0,2*np.pi,PLANE_DIRECTIONS,endpoint=False)
    return [np.cos(a)*first+np.sin(a)*second for a in angles]


def sample_queries(gp, f, rng, bounds):
    """Programs 3--6 for one posterior draw f; returns its start/goal pairs."""
    lo,hi=np.asarray(bounds,float).T
    dim=len(lo)
    signal=max(gp.signal,1e-10)
    pairs=[]
    def ray_bound(center,vector):
        return min([(hi[d]-center[d])/vector[d] if vector[d]>0 else
                    (lo[d]-center[d])/vector[d] for d in range(dim) if abs(vector[d])>1e-12])
    mi=solve_p3(gp,f,lo,hi)
    if mi is None: return pairs
    raw=f.derivative(mi)[0]
    norm=np.linalg.norm(raw)
    if norm<1e-10: return pairs
    center=mi+1.*raw  # Eq.7: eta=1; with unit-gradient estimates this offset is ~1 position unit.
    if np.any(center<lo) or np.any(center>hi) or f(center)[0]<=0: return pairs
    # Problem 4: the in-H direction along which the draw stays unsafe longest.
    p4=[]
    for vector in hyperplane_directions(raw):
        def unsafe(t): return f(center+np.asarray(t)[:,None]*vector)[:,None]/signal
        far=scalar_feasible_extreme(unsafe,ray_bound(center,vector),gp.ls,nearest=False)
        if far is not None: p4.append((far,far*vector))
    directions=[]
    if p4: directions.append((max(p4,key=lambda v:v[0])[1],(.55,.55)))
    directions.append((raw,(.30,.55)))
    # Problems 5 and 6: nearest safe endpoints (Eq. 10) on both sides.
    for direction,betas in directions:
        pair=[]
        for sign,beta in zip((-1,1),betas):
            vector=sign*direction/np.linalg.norm(direction)
            def safe(t):
                q=center+np.asarray(t)[:,None]*vector
                return endpoint_constraints(gp,f,q,beta)
            near=scalar_feasible_extreme(safe,ray_bound(center,vector),gp.ls)
            if near is None: break
            pair.append(center+near*vector)
        if len(pair)==2: pairs.append(pair)
    return pairs


def solve_queries(gp, rng, bounds, n_samples=None):
    """One Alg. 1 batch: P posterior draws, each giving Probs. 3--6 pairs."""
    pairs=[]
    for _ in range(n_samples or CONFIG['draws']):
        f=gp.sample_function(rng,1000,.3)
        try:
            pairs.extend(sample_queries(gp,f,rng,bounds))
        except (ValueError,ArithmeticError,np.linalg.LinAlgError,RuntimeError):
            # A failed local solve yields no query, as a failed IPOPT solve would.
            continue
    return pairs


def near_duplicate(pair, requested, tolerance=DUPLICATE_TOLERANCE):
    s,g=np.asarray(pair[0],float),np.asarray(pair[1],float)
    for s2,g2 in requested:
        if ((np.linalg.norm(s-s2)<tolerance and np.linalg.norm(g-g2)<tolerance) or
                (np.linalg.norm(s-g2)<tolerance and np.linalg.norm(g-s2)<tolerance)):
            return True
    return False


def sphere_directions(count):
    """Near-uniform unit vectors (Fibonacci lattice)."""
    i=np.arange(count)+.5
    z=1-2*i/count; r=np.sqrt(1-z*z); phi=np.pi*(1+5**.5)*i
    return np.column_stack([r*np.cos(phi),r*np.sin(phi),z])


def star_fill(points, query, unsafe, margin=.5, max_gap=np.pi/4):
    """Headroom variant: fill the deep interior enclosed by the tight states.

    Tight states, ordered by direction about their centroid, define a radial
    boundary (piecewise-linear in angle in 2D; nearest-direction average in
    3D). States at least `margin` inside it (only where observed directions
    are within `max_gap`) are also predicted unsafe, which corrects a
    stationary-kernel posterior mean that relaxes back to its safe prior
    inside large obstacles. No oracle labels or task parameters are used;
    only a star-shaped obstacle is assumed.
    """
    if len(points)<8: return unsafe
    centre=points.mean(axis=0)
    rel=points-centre
    q=np.asarray(query,float)-centre
    if points.shape[1]==3:
        radius=np.linalg.norm(rel,axis=1); direction=rel/np.maximum(radius,1e-12)[:,None]
        qr=np.linalg.norm(q,axis=1); qd=q/np.maximum(qr,1e-12)[:,None]
        fill=np.zeros(len(q),bool)
        for i in range(0,len(q),8192):
            cosine=np.clip(qd[i:i+8192]@direction.T,-1,1)
            nearest=np.argsort(-cosine,axis=1)[:,:4]
            angle=np.arccos(np.take_along_axis(cosine,nearest,axis=1))
            weight=1/(angle+1e-3)
            boundary=np.sum(weight*radius[nearest],axis=1)/np.sum(weight,axis=1)
            fill[i:i+8192]=(qr[i:i+8192]<boundary-margin)&(angle[:,-1]<=max_gap)
        return unsafe|fill
    angle=np.arctan2(rel[:,1],rel[:,0]); radius=np.linalg.norm(rel,axis=1)
    order=np.argsort(angle); angle=angle[order]; radius=radius[order]
    angle=np.r_[angle[-1]-2*np.pi,angle,angle[0]+2*np.pi]; radius=np.r_[radius[-1],radius,radius[0]]
    qa=np.arctan2(q[:,1],q[:,0]); qr=np.linalg.norm(q,axis=1)
    right=np.clip(np.searchsorted(angle,qa),1,len(angle)-1); left=right-1
    gap=angle[right]-angle[left]
    weight=np.where(gap>0,(qa-angle[left])/np.where(gap>0,gap,1.),0.)
    boundary=(1-weight)*radius[left]+weight*radius[right]
    return unsafe|((qr<boundary-margin)&(gap<=max_gap))


def send(value):
    print(json.dumps(value,allow_nan=False),flush=True)
    return json.loads(sys.stdin.readline())


def demonstrated_states(demos):
    return np.vstack([np.asarray(d['positions'],float) for d in demos])


def diagnostic(stage, demos, points, gp, **extra):
    # Unscored diagnostics only: the parent never uses these for metrics.
    info={'stage':stage,'demos':len(demos),'contacts':len(points),**extra}
    if gp is not None:
        info.update(lengthscale=float(gp.ls),signal=float(gp.signal),
                    noise=float(gp.noise),prior_mean=float(gp.constant))
    print(json.dumps(info,allow_nan=False),file=sys.stderr,flush=True)


class Observations:
    """Per-demonstration P7/P8 results, computed once per demonstration.

    App. A.1: P8's w recovers the constraint gradient only up to an unknown
    positive scale per state (it absorbs the true multiplier), and measured
    raw magnitudes scale with a demonstration's step length. The reference
    fixes this free scale by unit-norm estimates; raw=True keeps the
    multiplier-scaled magnitudes (an ablation).
    """
    def __init__(self, dynamics, raw=False):
        self.dynamics=dynamics; self.raw=raw; self.items=[]
    def __call__(self, demos):
        while len(self.items)<len(demos):
            self.items.append(kkt_tight_points(demos[len(self.items)],self.dynamics,CONFIG['tight_tol']))
        dim=self.dynamics.nc
        points=np.vstack([p for p,g in self.items]).reshape(-1,dim)
        gradients=np.vstack([g for p,g in self.items]).reshape(-1,dim)
        if not self.raw and len(gradients):
            gradients=gradients/np.linalg.norm(gradients,axis=1,keepdims=True)
        return points,gradients


def gp_acl_acquire(task, rng, demos, observe, topup=False):
    """Alg. 1, iterated until the demonstration or request budget is spent.

    Every iteration refits the GP to all demonstrations (Probs. 7-8 and the
    training loss), draws P posterior samples, solves Probs. 3-6 for each
    and requests the resulting start/goal pairs. Rejected (infeasible)
    requests return no demonstration, as an infeasible Prob. 1 would.
    topup=True (a variant, not the paper's method) spends demonstrations
    that Alg. 1 leaves unused on coverage crossings, then refits.
    """
    bounds=task['bounds']; budget=task['budget']; requests_left=task['query_limit']
    requested=[]; warm=None; batch=0
    while True:
        points,gradients=observe(demos)
        gp=train_gp(points,gradients,demonstrated_states(demos),init=warm)
        warm=None if gp is None else gp.z
        diagnostic(batch,demos,points,gp,requests=len(requested))
        if len(demos)>=budget or requests_left<=0 or batch>=MAX_BATCHES:
            break
        pairs=[]
        for _ in range(MAX_REDRAWS):
            if gp is None:
                lo,hi=np.asarray(bounds,float).T
                candidates=[rng.uniform(lo,hi,(2,len(lo))) for _ in range(2*CONFIG['draws'])]
            else:
                candidates=solve_queries(gp,rng,bounds)
            pairs=[pair for pair in candidates if not near_duplicate(pair,requested)]
            if pairs: break
        if not pairs:
            break
        for pair in pairs:
            if len(demos)>=budget or requests_left<=0: break
            if near_duplicate(pair,requested): continue
            response=send({'op':'query','start':np.asarray(pair[0]).tolist(),'goal':np.asarray(pair[1]).tolist()})
            requested.append((np.asarray(pair[0],float),np.asarray(pair[1],float)))
            requests_left=response.get('requests_remaining',0)
            if response.get('demo'):
                demos.append(response['demo'])
        batch+=1
    if topup and len(demos)<budget and requests_left>0:
        coverage_topup(task,demos,observe,requested,requests_left)
        points,gradients=observe(demos)
        gp=train_gp(points,gradients,demonstrated_states(demos),init=warm)
        diagnostic('topup',demos,points,gp,requests=len(requested))
    return gp


def coverage_topup(task, demos, observe, requested, requests_left):
    """Variant: spend unused demonstrations on crossings of uncovered arcs.

    About the tight-state centroid, the direction u farthest in angle from
    every observed tight state (and earlier targets) is chosen. A straight
    start/goal chord perpendicular to u, offset 0.6 median contact radii
    toward u, cuts the obstacle near that side, so the shortest feasible
    detour wraps around it there. No oracle labels or task parameters.
    """
    lo,hi=np.asarray(task['bounds'],float).T
    dim=len(lo); half=13/15*float(np.min(hi-lo))/2
    targeted=[]
    if dim==2:
        angles=np.linspace(-np.pi,np.pi,72,endpoint=False)
        candidates=np.column_stack([np.cos(angles),np.sin(angles)])
    else:
        candidates=sphere_directions(256)
    while len(demos)<task['budget'] and requests_left>0 and len(targeted)<len(candidates):
        points,_=observe(demos)
        if not len(points): return
        centre=points.mean(axis=0); rel=points-centre
        if dim==2:
            used=np.r_[np.arctan2(rel[:,1],rel[:,0]),np.asarray([np.arctan2(u[1],u[0]) for u in targeted],float)]
            gap=np.min(np.abs((angles[:,None]-used[None,:]+np.pi)%(2*np.pi)-np.pi),axis=1)
        else:
            used=np.vstack([rel/np.maximum(np.linalg.norm(rel,axis=1),1e-12)[:,None]]+[np.atleast_2d(u) for u in targeted])
            gap=np.arccos(np.clip(np.max(candidates@used.T,axis=1),-1,1))
        u=candidates[int(np.argmax(gap))]; targeted.append(u)
        radius=float(np.median(np.linalg.norm(rel,axis=1)))
        if dim==2:
            v=np.array([-u[1],u[0]])
        else:
            helper=np.eye(3)[int(np.argmin(np.abs(u)))]
            v=np.cross(u,helper); v/=np.linalg.norm(v)
        foot=centre+.6*radius*u
        pair=(np.clip(foot-half*v,lo+.5,hi-.5),np.clip(foot+half*v,lo+.5,hi-.5))
        if near_duplicate(pair,requested): continue
        response=send({'op':'query','start':pair[0].tolist(),'goal':pair[1].tolist()})
        requested.append(pair)
        requests_left=response.get('requests_remaining',0)
        if response.get('demo'):
            demos.append(response['demo'])


def random_acquire(task, rng, demos):
    """Random-sampling baseline: uniform start/goal pairs in the workspace."""
    lo,hi=np.asarray(task['bounds'],float).T
    requests_left=task['query_limit']
    while len(demos)<task['budget'] and requests_left>0:
        pair=rng.uniform(lo,hi,(2,len(lo)))
        response=send({'op':'query','start':pair[0].tolist(),'goal':pair[1].tolist()})
        requests_left=response.get('requests_remaining',0)
        if response.get('demo'):
            demos.append(response['demo'])


VARIANTS=['reference','random','no_acquisition','no_gradients','raw_gradients',
          'fill','topup','fill_topup','sweep']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='reference',choices=VARIANTS)
    parser.add_argument('--prior',default=CONFIG['prior'],choices=['zero','tied','learned'])
    parser.add_argument('--trust',type=float,default=CONFIG['trust'])
    parser.add_argument('--draws',type=int,default=CONFIG['draws'])
    parser.add_argument('--rho',type=float,default=CONFIG['rho'])
    parser.add_argument('--noise-floor',type=float,default=CONFIG['noise_floor'])
    parser.add_argument('--no-hinge',action='store_true')
    parser.add_argument('--tight-tol',type=float,default=CONFIG['tight_tol'])
    args=parser.parse_args()
    if not (np.isfinite(args.rho) and args.rho>0): parser.error('--rho must be positive')
    if not (0<args.noise_floor<1 and 0<args.tight_tol): parser.error('invalid noise floor or tolerance')
    CONFIG.update(prior=args.prior,trust=args.trust,
                  draws=args.draws,rho=args.rho,noise_floor=args.noise_floor,
                  hinge=not args.no_hinge,tight_tol=args.tight_tol)
    task=json.loads(sys.stdin.readline())
    rng=np.random.default_rng(task['seed'])
    demos=list(task['initial_demos'])
    dim=len(task['bounds'])
    dynamics=make_dynamics(task.get('system','double_integrator'),dim,task.get('dt',1.))
    observe=Observations(dynamics,raw=args.variant=='raw_gradients')
    if args.variant in ('random','no_acquisition','sweep'):
        if args.variant=='random':
            random_acquire(task,rng,demos)
        elif args.variant=='sweep':
            # A simple hand-written acquisition rule (not the paper's method):
            # coverage crossings aimed at the least-covered side from the start.
            coverage_topup(task,demos,observe,[],task['query_limit'])
        points,gradients=observe(demos)
        gp=train_gp(points,gradients,demonstrated_states(demos))
        diagnostic('final',demos,points,gp)
    else:
        # The reference acquisition; the remaining variants change only the
        # gradient observations or the final classifier.
        gp=gp_acl_acquire(task,rng,demos,observe,topup=args.variant in ('topup','fill_topup'))
    points,gradients=observe(demos)
    response=send({'op':'finish'})
    query=np.asarray(response['points'],float)
    unsafe=(gp.mean(query)>0) if gp is not None else np.zeros(len(query),bool)
    if args.variant=='no_gradients':
        ablation=train_gp(points,gradients,demonstrated_states(demos),True)
        pred=(ablation.mean(query)<=0).astype(int) if ablation else np.ones(len(query),int)
    elif args.variant in ('fill','fill_topup'):
        pred=(~star_fill(points,query,unsafe)).astype(int)
    else:
        pred=(~unsafe).astype(int)
    print(json.dumps({'predictions':pred.tolist()},allow_nan=False),flush=True)


if __name__=='__main__':
    main()
