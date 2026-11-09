"""Small meaningful checks of restored GP/KKT mathematical primitives."""
import numpy as np
from method.gp_acl import GPModel,kkt_tight_points,finite_gradient
p=np.array([[.4,.1],[-.5,.3]])
g=np.array([[.3,-.4],[-.2,.5]])
gp=GPModel(p,g,(.6,.7,1e-9))
assert np.max(np.abs(gp.mean_var(p)[0]))<1e-6
for point,gradient in zip(p,g):
    actual=finite_gradient(lambda q:gp.mean_var(q)[0],point)
    assert np.max(np.abs(actual-gradient))<1e-6
sample=gp.sample_function(np.random.default_rng(7),n_features=1000,alpha=0.)
assert np.max(np.abs(sample(p)-gp.mean_var(p)[0]))<1e-12
point,gradient=kkt_tight_points({'positions':[[0,0],[1,.2],[2,0]]})
assert len(point)==1 and np.allclose(gradient,[[0,-.8]])
point,gradient=kkt_tight_points({'positions':[[0,0],[1,0],[2,0]]})
assert len(point)==0
print('Derivative GP interpolation, posterior sample mean limit, and KKT programs: pass')
