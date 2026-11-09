"""Small arithmetic contract tests; no model fitting or private task solving."""
from evaluate import rank,load,HERE

# Direction is independent of coefficient sign; normalization removes units.
spec={'experiments':{'x':{'weight':1.,'metrics':{'a':{'coefficient':-2.,'higher_is_better':True},'b':{'coefficient':-3.,'higher_is_better':False}}}}}
anchor={'x':{'a':.5,'b':10.}}
cohort={'reference':{'x':{'a':.5,'b':10.}},'better':{'x':{'a':.7,'b':9.}},'failed':{'x':{'a':1.,'b':0.}}}
first=rank(cohort,anchor,{'failed':['constraint']},spec)
assert first['values']['better']>0 and first['values']['reference']==0
assert first['values']['failed'] is None and first['order'][-1]=='failed'
rescaled={name:{'x':{'a':v['x']['a']*1000,'b':v['x']['b']*.001}} for name,v in cohort.items()}
second=rank(rescaled,{'x':{'a':500,'b':.01}}, {'failed':['constraint']},spec)
assert abs(first['values']['better']-second['values']['better'])<1e-10
zero=rank({'r':anchor},anchor,{},spec)
assert zero['values']['r']==0
empty=rank({'failed':anchor},anchor,{'failed':['constraint']},spec)
assert empty['values']['failed'] is None
assert all(value==0 for value in empty['population_std'].values())
print('Ranking direction, coefficient magnitudes, unit invariance, zero variance and failure ordering: pass')
