"""Small meaningful checks of restored GP/KKT mathematical primitives."""
import numpy as np
from method.gp_acl import GPModel,kkt_tight_points,finite_gradient,scalar_feasible_extreme,marginal_likelihood
p=np.array([[.4,.1],[-.5,.3]])
g=np.array([[.3,-.4],[-.2,.5]])
gp=GPModel(p,g,(.6,.7,1e-9))
# Verify the actual training objective, including all derivative covariances.
for values_only in (False,True):
    logp=np.log([.6,.7,.003])
    value,analytic=marginal_likelihood(logp,p,g,values_only)
    step=np.eye(3)*1e-5
    numeric=np.array([(marginal_likelihood(logp+s,p,g,values_only)[0]
                      -marginal_likelihood(logp-s,p,g,values_only)[0])/2e-5 for s in step])
    assert np.allclose(analytic,numeric,rtol=1e-5,atol=1e-6),(analytic,numeric)
assert np.max(np.abs(gp.mean_var(p)[0]))<1e-6
for point,gradient in zip(p,g):
    actual=finite_gradient(lambda q:gp.mean_var(q)[0],point)
    assert np.max(np.abs(actual-gradient))<1e-6
sample=gp.sample_function(np.random.default_rng(7),n_features=1000,alpha=0.)
assert np.max(np.abs(sample(p)-gp.mean_var(p)[0]))<1e-12
for point in p:
    dm,dv=gp.derivatives(point)
    assert np.max(np.abs(dm[0]-finite_gradient(lambda q:gp.mean_var(q)[0],point)))<1e-6
    assert np.max(np.abs(dv[0]-finite_gradient(lambda q:gp.mean_var(q)[1],point)))<1e-6
sample=gp.sample_function(np.random.default_rng(9))
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
print('GP likelihood/posterior analytic derivatives, continuous ray programs, and KKT programs: pass')
