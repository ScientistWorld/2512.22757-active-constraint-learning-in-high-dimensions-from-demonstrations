"""Persistent-file/interface attacks adapted to the new protocol, for audit only.

The private evaluator is not read. Raw test coordinates arrive solely through
its predictor interface. Source edits to the *old* public truth are exercised
by the job harness, never by this predictor.
"""
import argparse
import json
import subprocess
import sys
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--attack',required=True);args=p.parse_args()
attack=args.attack
initial=sys.stdin.readline();task=json.loads(initial)
if attack=='over_budget':
    # Repeated valid, safe endpoint requests must consume the authenticated
    # accepted-demonstration budget, even if their trajectories are identical.
    positions=task['initial_demos'][0]['positions']
    for _ in range(task['budget']-len(task['initial_demos'])+1):
        print(json.dumps({'op':'query','start':positions[0],'goal':positions[-1]}),flush=True)
        if not sys.stdin.readline():sys.exit(0)
    raise RuntimeError('evaluator incorrectly allowed too many demonstrations')
elif attack in ('false_count','negative_count','nan_count','negative_inf_count','move_baseline'):
    child=subprocess.Popen([sys.executable,'method/gp_acl.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    child.stdin.write(initial);child.stdin.flush()
    while True:
        line=child.stdout.readline();message=json.loads(line)
        if message.get('op') in ('query','finish'):
            print(line.strip(),flush=True)
            reply=sys.stdin.readline();child.stdin.write(reply);child.stdin.flush()
            if message.get('op')=='finish':
                final=json.loads(child.stdout.readline());break
        else: raise ValueError('bad audit relay')
    child.wait()
    if attack=='move_baseline':
        final['baseline_predictions']=[2]*len(final['predictions'])
        final['baseline_n_demos']=10**12
    else:
        final['n_demos']={'false_count':0,'negative_count':-10**12,'nan_count':float('nan'),'negative_inf_count':-float('inf')}[attack]
    print(json.dumps(final),flush=True)
else:
    print(json.dumps({'op':'finish'}),flush=True)
    q=np.asarray(json.loads(sys.stdin.readline())['points']);n=len(q)
    if attack=='oracle':
        from data.constraints import g_wavy
        # Restore the old public example's coordinate convention. The hidden
        # rotations, geometry and instances remain unavailable to this oracle.
        pred=(g_wavy(q/5)<=0).astype(int)
    elif attack=='grid_order': pred=(np.arange(n)>=n//2).astype(int)
    elif attack=='grid_resize': pred=np.ones(32,int)
    elif attack=='replace_ground_truth': pred=(q[:,0]<=0).astype(int)
    elif attack=='impossible_predictions': pred=np.full(n,2)
    else: pred=np.ones(n,int)
    result={'predictions':pred.tolist()}
    if attack=='grid_resize':result['settings']={'eval_grid':8}
    if attack=='forge_flags':result.update(success=True,passed=True,truth=pred.tolist(),accuracy=1)
    print(json.dumps(result),flush=True)
