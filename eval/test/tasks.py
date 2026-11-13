"""Private task parameters, answer generation, and demonstration oracle.

This file and its seeds stay with eval/test. Nothing is imported from data/,
method/, baseline/, or the training evaluator. A continuous infinity of unseen
instances replaces the public finite answer grid.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

TASK_SEED = 94107231
TASKS_PER_EXPERIMENT = 36
N_UNIFORM = 4096
N_BOUNDARY = 8192
HORIZON = 30
# Alg. 1 takes an initial demonstration set S_D^(1) as input. The paper's
# runs start from a weak first model (Fig. 5), and its random-query baseline
# ends with only the parts of g3 its first demonstrations touched (Fig. 8):
# the initial set covers the obstacle only partially. Every method shares the
# first INITIAL_CROSSINGS fixed crossings below (selected on the development
# split, scripts/select_setup.py) and then acquires demonstrations up to the
# total BUDGET, which includes the shared initial ones.
BUDGET = 12
REQUEST_LIMIT = 40
# The paper's 2D double-integrator g3 figures use the workspace [-15, 15]^2.
BOUND = 15.0
# Obstacles are translated so a fixed origin-centred shape is not an answer.
CENTER_RANGE = 2.0
BAND_WIDTH = .5
NEAR_WIDTH = .175
EXPERIMENTS = ('double_integrator_g3', 'complex_boundary_stress')
# Fixed world-coordinate endpoints, outside every instance, shared by methods.
# Each crossing passes the workspace centre, so its expert detour wraps one
# side of every instance.
CROSSINGS = (([-13.5,-11.5],[13.5,11.5]),([-13.5,11.5],[13.5,-11.5]),
             ([-14.,1.5],[14.,-1.5]),([1.5,14.],[-1.5,-14.]))
INITIAL_CROSSINGS = 4
INITIAL_PAIRS = CROSSINGS[:INITIAL_CROSSINGS]
# Private per-instance smooth radial deformation of the paper's g3 polynomial:
# localized angular bumps and dents (von Mises profiles). Its form and size
# stay here; the development split uses a different family of comparable
# size, so exact public obstacle formulas do not answer hidden instances.
DEFORM_BUMPS = 3
DEFORM_AMPLITUDE = .10
DEFORM_CONCENTRATION = 3.
# Private complex-boundary family: two distinct dominant low harmonics per
# instance from this set, with slope-matched amplitudes 0.5*s/h, s~U(.75,1.25),
# plus the random high-harmonic spectrum. (The development family fixes
# harmonics 3 and 5 at 0.16 and 0.10, the same 0.5/h slope envelope, so a
# hand-written development generator does not answer hidden instances.)
STRESS_LOW_HARMONICS = (3, 4, 5)
STRESS_SLOPE = .5


def g3_root(t):
    """Radius of the paper's g3 boundary along shape-frame angle t."""
    co=np.cos(t); si=np.sin(t)
    a=.25*co**2+si**2
    b=-50*a+50*co**2
    return np.sqrt((-b+np.sqrt(b*b+250*a*a))/(2*a*a))


class Task:
    def __init__(self, experiment, index):
        self.experiment=experiment; self.index=index
        rng=np.random.default_rng(np.random.SeedSequence([TASK_SEED,EXPERIMENTS.index(experiment),index]))
        self.rotation=rng.uniform(-np.pi,np.pi)
        self.scale=rng.uniform(.88,1.08)
        self.phase=rng.uniform(-np.pi,np.pi,21)
        self.coefficients=rng.normal(0,.035,15)*(6/np.arange(6,21))**1.3
        self.center=rng.uniform(-CENTER_RANGE,CENTER_RANGE,2)
        self.deform=rng.uniform(-DEFORM_AMPLITUDE,DEFORM_AMPLITUDE,DEFORM_BUMPS)
        self.deform_phase=rng.uniform(-np.pi,np.pi,DEFORM_BUMPS)
        # Separate stream: leaves every other parameter and all evaluation
        # coordinates of the main stream unchanged.
        extra=np.random.default_rng(np.random.SeedSequence([TASK_SEED,EXPERIMENTS.index(experiment),index,1]))
        harmonics=extra.choice(STRESS_LOW_HARMONICS,2,replace=False)
        self.low=[(int(h),STRESS_SLOPE*extra.uniform(.75,1.25)/h) for h in harmonics]
        self.rng=rng

    def warp(self, t):
        """Radial scale w(t)=scale*(1+d(t)) of the g3 shape and dw/dt."""
        t=np.asarray(t,float)
        d=np.zeros_like(t); dd=np.zeros_like(t)
        for amplitude,centre in zip(self.deform,self.deform_phase):
            bump=amplitude*np.exp(DEFORM_CONCENTRATION*(np.cos(t-centre)-1))
            d=d+bump; dd=dd-DEFORM_CONCENTRATION*np.sin(t-centre)*bump
        return self.scale*(1+d),self.scale*dd

    def radius(self, theta):
        theta=np.asarray(theta)
        t=theta-self.rotation
        if self.experiment==EXPERIMENTS[0]:
            return self.warp(t)[0]*g3_root(t)
        r=1.03+sum(a*np.sin(h*t+self.phase[h]) for h,a in self.low)
        for j,h in enumerate(range(6,21)):
            r=r+self.coefficients[j]*np.sin(h*t+self.phase[h])
        return 5*self.scale*r

    def boundary_distance(self, points):
        """Radial distance to the boundary about the obstacle centre."""
        q=np.atleast_2d(points)-self.center
        return np.abs(np.linalg.norm(q,axis=1)-self.radius(np.arctan2(q[:,1],q[:,0])))

    def value_gradient(self, points):
        p=np.atleast_2d(points)-self.center
        if self.experiment==EXPERIMENTS[0]:
            # The paper's g3 polynomial in radially warped shape coordinates
            # z=z0/w(t), z0 the rotated offset and t its polar angle.
            co,si=np.cos(self.rotation),np.sin(self.rotation)
            rotation=np.array([[co,si],[-si,co]])
            z0=p@rotation.T
            t=np.arctan2(z0[:,1],z0[:,0])
            w,dw=self.warp(t)
            z=z0/w[:,None]
            x,y=z.T; a=.25*x*x+y*y-25
            value=(-a*a-50*x*x+687.5)/687.5
            gz=np.column_stack([-a*x-100*x,-4*a*y])/687.5
            r2=np.maximum(np.sum(z0**2,axis=1),1e-300)
            dt=np.column_stack([-z0[:,1],z0[:,0]])/r2[:,None]
            g0=gz/w[:,None]-(np.sum(gz*z0,axis=1)*dw/w**2)[:,None]*dt
            return value,g0@rotation
        theta=np.arctan2(p[:,1],p[:,0]); rr=np.maximum(np.linalg.norm(p,axis=1),1e-8)
        t=theta-self.rotation
        radius=self.radius(theta)
        deriv=sum(a*h*np.cos(h*t+self.phase[h]) for h,a in self.low)
        for j,h in enumerate(range(6,21)):
            deriv+=h*self.coefficients[j]*np.cos(h*t+self.phase[h])
        deriv*=5*self.scale
        angular=np.column_stack([-p[:,1],p[:,0]])/rr[:,None]**2
        value=radius**2-rr**2
        grad=2*radius[:,None]*deriv[:,None]*angular-2*p
        return value,grad

    def evaluation(self):
        # The uniform part preserves the paper's quantity; an independent,
        # shuffled narrow boundary band gives statistically useful resolution.
        uniform=self.rng.uniform(-BOUND,BOUND,(N_UNIFORM,2))
        angle=self.rng.uniform(-np.pi,np.pi,N_BOUNDARY)
        offset=self.rng.uniform(-BAND_WIDTH,BAND_WIDTH,N_BOUNDARY)
        radius=self.radius(angle)+offset
        band=self.center+np.column_stack([radius*np.cos(angle),radius*np.sin(angle)])
        points=np.vstack([uniform,band]); is_uniform=np.arange(len(points))<N_UNIFORM
        order=self.rng.permutation(len(points))
        points=points[order]
        truth=(self.value_gradient(points)[0]<=0).astype(int)
        return points,truth,is_uniform[order]

    def initial_demonstrations(self):
        # Endpoints are fixed in world coordinates and guaranteed outside all
        # instances. They carry no obstacle parameters or selected contacts.
        # Shared initial observations count against every method's total cap.
        result=[self.demonstrate(np.asarray(s,float),np.asarray(g,float)) for s,g in INITIAL_PAIRS]
        if any(d is None for d in result):
            raise RuntimeError('Fixed initial demonstration failed')
        return result

    def demonstrate(self,start,goal):
        if np.any(self.value_gradient(np.vstack([start,goal]))[0]>1e-8) or np.linalg.norm(start-goal)<.15:
            return None
        line=np.linspace(start,goal,HORIZON)
        def unpack(z): return np.vstack([start,z.reshape(-1,2),goal])
        def cost(z): return np.sum(np.diff(unpack(z),axis=0)**2)
        def jac(z):
            p=unpack(z); return (2*(2*p[1:-1]-p[:-2]-p[2:])).ravel()
        def constraint(z): return -self.value_gradient(unpack(z)[1:-1])[0]
        def constraint_jac(z):
            grad=-self.value_gradient(unpack(z)[1:-1])[1]
            mat=np.zeros((HORIZON-2,2*(HORIZON-2)))
            k=np.arange(HORIZON-2); mat[k,2*k]=grad[:,0]; mat[k,2*k+1]=grad[:,1]
            return mat
        positions=None
        if np.min(constraint(line[1:-1].ravel()))>=0:
            # The feasible straight line is the exact unconstrained minimizer.
            positions=line
        else:
            # P1 permits a locally optimal expert. Start from a feasible route
            # around the obstacle, rather than competing infeasible starts
            # whose cheapest solution can jump across a narrow waist at knots.
            # These are initial guesses only: the objective and discrete
            # constraints solved below are exactly the same P1 program.
            endpoints=np.vstack([start,goal])-self.center
            start_angle,goal_angle=np.arctan2(endpoints[:,1],endpoints[:,0])
            turn=(goal_angle-start_angle+np.pi)%(2*np.pi)-np.pi
            alternate=turn-2*np.pi if turn>=0 else turn+2*np.pi
            fraction=np.linspace(0,1,HORIZON)
            endpoint_radius=np.linspace(*np.linalg.norm(endpoints,axis=1),HORIZON)
            for angular_turn in (turn,alternate):
                angles=start_angle+angular_turn*fraction
                radius=np.maximum(endpoint_radius,self.radius(angles)+.3)
                warm=self.center+np.column_stack([radius*np.cos(angles),radius*np.sin(angles)])
                warm[0]=start;warm[-1]=goal
                result=minimize(cost,warm[1:-1].ravel(),jac=jac,method='SLSQP',
                                constraints=[{'type':'ineq','fun':constraint,'jac':constraint_jac}],
                                options={'maxiter':400,'ftol':2e-12})
                if result.success and np.min(constraint(result.x))>=-1e-7:
                    positions=unpack(result.x)
                    break
        if positions is None:
            return None
        velocity=np.vstack([np.diff(positions,axis=0),positions[-1]-positions[-2]])
        controls=np.diff(velocity,axis=0)
        return {'positions':positions.tolist(),'velocities':velocity.tolist(),'controls':controls.tolist()}
