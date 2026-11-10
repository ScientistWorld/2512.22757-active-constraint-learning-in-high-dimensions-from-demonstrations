"""Small meaningful checks of restored GP/KKT mathematical primitives."""
import numpy as np
from method.gp_acl import (GPModel,kkt_tight_points,finite_gradient,scalar_feasible_extreme,
                           marginal_likelihood,training_objective,train_gp,endpoint_constraints,observations)
p=np.array([[.4,.1],[-.5,.3]])
g=np.array([[.3,-.4],[-.2,.5]])
gp=GPModel(p,g,(.6,.7,1e-9))
# Verify the likelihood term, including all derivative covariances.
for values_only in (False,True):
    logp=np.log([.6,.7,.003])
    value,analytic=marginal_likelihood(logp,p,g,values_only)
    step=np.eye(3)*1e-5
    numeric=np.array([(marginal_likelihood(logp+s,p,g,values_only)[0]
                      -marginal_likelihood(logp-s,p,g,values_only)[0])/2e-5 for s in step])
    assert np.allclose(analytic,numeric,rtol=1e-5,atol=1e-6),(analytic,numeric)
# Verify the full Sec. IV-B training loss (likelihood + feasibility hinge) in
# all four variables, at states where the hinge is both active and inactive.
states=np.array([[1.3,.2],[-1.1,.9],[.1,-1.4],[2.5,2.5],[.05,.2],[-.3,-.2]])
for values_only in (False,True):
    for z in (np.r_[np.log([.6,.7,.003]),-.4],np.r_[np.log([.9,.4,.01]),.15]):
        value,analytic=training_objective(z,p,g,states,values_only)
        step=np.eye(4)*1e-6
        numeric=np.array([(training_objective(z+s,p,g,states,values_only)[0]
                          -training_objective(z-s,p,g,states,values_only)[0])/2e-6 for s in step])
        assert np.allclose(analytic,numeric,rtol=1e-4,atol=1e-6),(values_only,z,analytic,numeric)
# A constant prior mean leaves the observed zero values and gradients intact.
shifted=GPModel(p,g,(.6,.7,1e-9),constant=-.8)
assert np.max(np.abs(gp.mean_var(p)[0]))<1e-6
assert np.max(np.abs(shifted.mean_var(p)[0]))<1e-6
assert abs(shifted.mean_var(np.array([[40.,40.]]))[0][0]+.8)<1e-9
for model in (gp,shifted):
    for point,gradient in zip(p,g):
        actual=finite_gradient(lambda q:model.mean_var(q)[0],point)
        assert np.max(np.abs(actual-gradient))<1e-6
    sample=model.sample_function(np.random.default_rng(7),n_features=1000,alpha=0.)
    assert np.max(np.abs(sample(p)-model.mean_var(p)[0]))<1e-12
    for point in p:
        dm,dv=model.derivatives(point)
        assert np.max(np.abs(dm[0]-finite_gradient(lambda q:model.mean_var(q)[0],point)))<1e-6
        assert np.max(np.abs(dv[0]-finite_gradient(lambda q:model.mean_var(q)[1],point)))<1e-6
    sample=model.sample_function(np.random.default_rng(9))
    for point in p:
        assert np.max(np.abs(sample.derivative(point)[0]-finite_gradient(sample,point)))<1e-6
# P4 must find the furthest disconnected feasible interval; P5 must find the
# nearest intersection of two constraints, not a fixed candidate's position.
near=scalar_feasible_extreme(lambda t:np.column_stack([t-.314159,1.7-t]),2.,.5)
assert abs(near-.314159)<1e-6
far=scalar_feasible_extreme(lambda t:((t-.2)*(t-.8)*(t-1.3)*(1.9-t))[:,None],2.,.5,False)
assert abs(far-1.9)<1e-6
point,gradient=kkt_tight_points({'positions':[[0,0],[1,.2],[2,0]]})
assert len(point)==1 and np.allclose(gradient,[[0,-.8]])
point,gradient=kkt_tight_points({'positions':[[0,0],[1,0],[2,0]]})
assert len(point)==0
# P8 estimates are identified up to a positive scale: unit directions are kept.
bent=[{'positions':[[0,0],[1,.2],[2,0]]},{'positions':[[0,0],[.1,.02],[.2,0]]}]
point,gradient=observations(bent)
assert len(point)==2 and np.allclose(gradient,[[0,-1],[0,-1]])
point,gradient=observations(bent,raw=True)
assert np.allclose(gradient,[[0,-.8],[0,-.08]])
print('GP likelihood, hinge training loss, constant-mean posterior derivatives, continuous ray programs, and KKT programs: pass',flush=True)

# Eq. 10 as a safety probability under the paper's avoid set {g>0}:
# Phi(-mean/sd)>=beta, plus the separate sampled-function margin.
class ControlledPosterior:
    signal=2.
    def mean_var(self,query):
        return np.array([-1.,1.]),np.ones(2)
constraints=endpoint_constraints(ControlledPosterior(),lambda q:np.array([-.003,.003]),
                                np.zeros((2,2)),.55)
assert np.allclose(constraints[:,0],[.001,-.002])
assert np.allclose(constraints[:,1],[.2913447460685429,-.3913447460685429])
print('Eq. 10 safe-probability CDF and independent sampled-function constraint: pass',flush=True)

# Unscored behavioural probe: inward gradients on a ring of contacts with
# demonstrated states outside it. Report the fitted model; no assertion.
angles=np.linspace(0,2*np.pi,13)[:-1]
ring=3*np.column_stack([np.cos(angles),np.sin(angles)])
model=train_gp(ring,-.5*ring/3,4.5*np.column_stack([np.cos(angles+.2),np.sin(angles+.2)]))
probe=np.array([[0.,0.],[1.5,0.],[3.5,0.],[6.,0.],[12.,12.]])
print('Ring probe: length %.3f signal %.3f noise %.2e prior mean %.3f; mean at r=0,1.5,3.5,6,17: %s'%(
    model.ls,model.signal,model.noise,model.constant,np.round(model.mean_var(probe)[0],3).tolist()),flush=True)
