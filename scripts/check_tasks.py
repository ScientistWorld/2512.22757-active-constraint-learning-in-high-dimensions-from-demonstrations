"""Job-only checks of both task generators (no method code is imported).

Finite-difference checks of the constraint gradients the experts use, zero
constraint value on the generators' boundary radius, and a comparison of the
development and hidden deformation sizes (comparable difficulty).
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(side):
    spec=importlib.util.spec_from_file_location(f'{side}_tasks',ROOT/'eval'/side/'tasks.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


report={}
for side in ('train','test'):
    tasks=load(side)
    rng=np.random.default_rng(3)
    deform=[]
    for experiment in tasks.EXPERIMENTS:
        worst_gradient=0.;worst_boundary=0.
        for index in range(4):
            task=tasks.Task(experiment,index)
            angles=rng.uniform(-np.pi,np.pi,200)
            for factor in (.5,.9,1.1,1.6):
                radius=task.radius(angles)*factor
                points=task.center+np.column_stack([radius*np.cos(angles),radius*np.sin(angles)])
                value,gradient=task.value_gradient(points)
                numeric=np.zeros_like(gradient)
                for d in range(2):
                    step=np.zeros(2);step[d]=1e-6
                    numeric[:,d]=(task.value_gradient(points+step)[0]-task.value_gradient(points-step)[0])/2e-6
                scale=np.maximum(np.linalg.norm(gradient,axis=1),1e-3)
                worst_gradient=max(worst_gradient,float(np.max(np.linalg.norm(numeric-gradient,axis=1)/scale)))
                # Inside the boundary radius is unsafe (value > 0), outside safe.
                assert np.all((value>0)==(factor<1)),(side,experiment,index,factor)
            boundary=task.center+task.radius(angles)[:,None]*np.column_stack([np.cos(angles),np.sin(angles)])
            value,gradient=task.value_gradient(boundary)
            worst_boundary=max(worst_boundary,float(np.max(np.abs(value)/np.linalg.norm(gradient,axis=1))))
        if experiment==tasks.EXPERIMENTS[0]:
            grid=np.linspace(-np.pi,np.pi,721)
            for index in range(tasks.TASKS_PER_EXPERIMENT):
                task=tasks.Task(experiment,index)
                deform.append(np.sqrt(np.mean((task.warp(grid)[0]/task.scale-1)**2)))
        assert worst_gradient<1e-4,(side,experiment,worst_gradient)
        assert worst_boundary<1e-6,(side,experiment,worst_boundary)
        report[f'{side}.{experiment}']={'max_relative_gradient_error':worst_gradient,'max_boundary_offset':worst_boundary}
    report[f'{side}.g3_deformation_rms']=float(np.mean(deform))
print('Task generator gradient, boundary and deformation checks: pass',report,flush=True)
