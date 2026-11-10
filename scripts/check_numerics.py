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

# New Problem 3 machinery: zero-crossing enumeration, projection, the
# trust-region global solve, duplicate requests, and memoized/warm training.
from method.gp_acl import (boundary_crossings,nearest_distance,project_to_boundary,solve_p3,
                           near_duplicate,star_fill,solve_queries,TRUST_LENGTHSCALES)
def circle(q):
    q=np.atleast_2d(q);return 4.-np.sum(q**2,axis=1)
circle.derivative=lambda q:-2*np.atleast_2d(q)
crossings=boundary_crossings(circle,np.array([-3.,-3.]),np.array([3.,3.]),.1)
assert len(crossings)>100 and np.max(np.abs(np.linalg.norm(crossings,axis=1)-2.))<.01
assert np.allclose(nearest_distance(np.array([[0.,0.],[3.,4.]]),np.array([[0.,1.],[3.,0.]])),[1.,4.])
projected=project_to_boundary(circle,np.array([1.,1.]),1.)
assert projected is not None and abs(np.linalg.norm(projected)-2.)<1e-8
assert near_duplicate((np.array([0.,0.]),np.array([1.,1.])),[(np.array([1.05,1.]),np.array([0.,.1]))])
assert not near_duplicate((np.array([0.,0.]),np.array([1.,1.])),[(np.array([2.,1.]),np.array([0.,0.]))])
ring_states=4.5*np.column_stack([np.cos(angles+.2),np.sin(angles+.2)])
first=train_gp(ring,-ring/3,ring_states)
again=train_gp(ring,-ring/3,ring_states)
assert np.allclose(first.z,again.z)
warm=train_gp(ring,-ring/3,ring_states,init=first.z)
assert np.all(np.isfinite(warm.z))
# Half-ring of contacts: P3 must leave the observations (zero variance
# gradient there) and return a sampled-boundary state of higher variance,
# within the trust region, for most posterior draws.
half=ring[:6];half_gp=train_gp(half,-half/3,ring_states[:6])
contact_variance=float(np.max(half_gp.mean_var(half)[1]))
# Variance midway between observed contacts on the arc (low: inside the data).
gap_angles=angles[:5]+np.pi/12
gap_variance=float(np.max(half_gp.mean_var(3*np.column_stack([np.cos(gap_angles),np.sin(gap_angles)]))[1]))
rng=np.random.default_rng(11);found=0;ratios=[]
for _ in range(6):
    sample=half_gp.sample_function(rng,1000,.3)
    point=solve_p3(half_gp,sample,np.array([-15.,-15.]),np.array([15.,15.]))
    if point is None: continue
    found+=1
    assert abs(sample(point)[0])/half_gp.signal<1e-5
    assert nearest_distance(point[None],half)[0]<=TRUST_LENGTHSCALES*half_gp.ls+1.
    variance=half_gp.mean_var(point)[1][0]
    assert variance>10*contact_variance,(variance,contact_variance)
    ratios.append(float(variance/max(gap_variance,1e-300)))
assert found>=4,found
pairs=solve_queries(half_gp,np.random.default_rng(5),[[-15,15],[-15,15]])
assert all(np.all(np.isfinite(np.asarray(p))) for p in pairs)
# Interior fill marks only states well inside the tight-state polygon.
fill=star_fill(ring,np.array([[0.,0.],[2.9,0.],[5.,0.]]),np.zeros(3,bool))
assert fill.tolist()==[True,False,False]
print('Problem 3 crossing enumeration, projection, trust-region solve, duplicate filter, memoized fits and fill: pass;',
      'half-ring P3 solutions',found,'of 6; batch pairs',len(pairs),
      '; fitted length %.3f signal %.3f; P3 variance / arc-gap variance'%(half_gp.ls,half_gp.signal),
      np.round(ratios,2).tolist(),flush=True)
