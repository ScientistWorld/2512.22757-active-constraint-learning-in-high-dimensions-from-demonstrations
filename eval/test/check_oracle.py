"""Job-only checks of expert feasibility, dynamics and P1 stationarity."""
import numpy as np
from tasks import Task,EXPERIMENTS

diagnostics={}
for experiment in EXPERIMENTS:
    maximum_gradient=0.;maximum_stationarity=0.
    for index in (0,1):
        task=Task(experiment,index)
        demonstrations=task.initial_demonstrations()
        assert len(demonstrations)==4
        for demonstration in demonstrations:
            p=np.asarray(demonstration['positions'])
            v=np.asarray(demonstration['velocities'])
            u=np.asarray(demonstration['controls'])
            assert p.shape==(30,2) and np.all(np.isfinite(p))
            assert np.allclose(np.diff(p,axis=0),v[:-1],atol=1e-12)
            assert np.allclose(np.diff(v,axis=0),u,atol=1e-12)
            values,gradients=task.value_gradient(p[1:-1])
            assert np.max(values)<=1e-7
            objective_gradient=2*(2*p[1:-1]-p[:-2]-p[2:])
            # Independent projection onto the nonnegative multiplier cone.
            multipliers=np.maximum(0,-np.sum(objective_gradient*gradients,axis=1)
                                     /np.sum(gradients**2,axis=1))
            residual=np.max(np.abs(objective_gradient+multipliers[:,None]*gradients))
            assert residual<1e-3,residual
            assert np.max(np.abs(multipliers*values))<1e-5
            maximum_gradient=max(maximum_gradient,float(np.max(np.linalg.norm(objective_gradient,axis=1))))
            maximum_stationarity=max(maximum_stationarity,float(residual))
        # An unobstructed straight trajectory has the known exact optimum.
        start=np.array([9.5,-8.]);goal=np.array([9.5,8.])
        straight=task.demonstrate(start,goal)
        assert straight is not None
        assert np.allclose(straight['positions'],np.linspace(start,goal,30),atol=1e-12)
    diagnostics[experiment]={'maximum_initial_gradient':maximum_gradient,
                             'maximum_stationarity_residual':maximum_stationarity}
print('Expert feasibility, eliminated dynamics and P1 KKT checks: pass',diagnostics,flush=True)
