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
# Sparse-demonstration regime: two shared initial demonstrations plus the
# two Alg. 1 acquisition stages of four demonstrations each.
BUDGET = 10
REQUEST_LIMIT = 40
# The paper's 2D double-integrator g3 figures use the workspace [-15, 15]^2.
BOUND = 15.0
# Obstacles are translated so a fixed origin-centred shape is not an answer.
CENTER_RANGE = 2.0
BAND_WIDTH = .5
NEAR_WIDTH = .175
EXPERIMENTS = ('double_integrator_g3', 'complex_boundary_stress')
# Fixed world-coordinate endpoints, outside every instance, shared by methods.
INITIAL_PAIRS = (([-13.5,-11.5],[13.5,11.5]),([-13.5,11.5],[13.5,-11.5]))


class Task:
    def __init__(self, experiment, index):
        self.experiment=experiment; self.index=index
        rng=np.random.default_rng(np.random.SeedSequence([TASK_SEED,EXPERIMENTS.index(experiment),index]))
        self.rotation=rng.uniform(-np.pi,np.pi)
        self.scale=rng.uniform(.88,1.08)
        self.phase=rng.uniform(-np.pi,np.pi,21)
        self.coefficients=rng.normal(0,.035,15)*(6/np.arange(6,21))**1.3
        self.center=rng.uniform(-CENTER_RANGE,CENTER_RANGE,2)
        self.rng=rng

    def radius(self, theta):
        theta=np.asarray(theta)
        if self.experiment==EXPERIMENTS[0]:
            t=theta-self.rotation
            co=np.cos(t); si=np.sin(t)
            a=.25*co**2+si**2
            b=-50*a+50*co**2
            r2=(-b+np.sqrt(b*b+250*a*a))/(2*a*a)
            return self.scale*np.sqrt(r2)
        t=theta-self.rotation
        r=1.03+.16*np.sin(3*t+self.phase[3])+.10*np.cos(5*t+self.phase[5])
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
            co,si=np.cos(self.rotation),np.sin(self.rotation)
            rotation=np.array([[co,si],[-si,co]])
            z=p@rotation.T/self.scale
            x,y=z.T; a=.25*x*x+y*y-25
            value=(-a*a-50*x*x+687.5)/687.5
            grad=np.column_stack([-a*x-100*x,-4*a*y])@rotation/self.scale/687.5
            return value,grad
        theta=np.arctan2(p[:,1],p[:,0]); rr=np.maximum(np.linalg.norm(p,axis=1),1e-8)
        t=theta-self.rotation
        radius=self.radius(theta)
        deriv=.48*np.cos(3*t+self.phase[3])-.5*np.sin(5*t+self.phase[5])
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
