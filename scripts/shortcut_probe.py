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

p=argparse.ArgumentParser();p.add_argument('--attack',required=True)
p.add_argument('--prior',default=None,help='prior flag passed to the relayed reference')
p.add_argument('--rho',default=None,help='rho flag passed to the relayed reference')
args=p.parse_args()
attack=args.attack
RELAY=([sys.executable,'method/gp_acl.py']+(['--prior',args.prior] if args.prior else [])
       +(['--rho',args.rho] if args.rho else []))


def fit_stress_family(contacts):
    """MAP fit of the *visible development* complex-boundary generator.

    r(theta)=R*(1.03+.16 sin(3 theta+a)+.10 cos(5 theta+b)+sum_h b_h sin(h theta)
    +c_h cos(h theta)), h=6..20, with the generator's own Gaussian prior on the
    high harmonics and its scale range, fitted to every tight state. Returns
    (parameters, RMS radial residual in position units, radius function).
    """
    from scipy.optimize import least_squares
    H=np.arange(6,21);sigma=.035*(6/H)**1.3/np.sqrt(2)
    def radius(theta,z):
        theta=np.asarray(theta,float)
        r=1.03+.16*np.sin(3*theta+z[3])+.10*np.cos(5*theta+z[4])
        if len(z)>5:
            r=r+np.sin(np.outer(theta,H))@z[5:20]+np.cos(np.outer(theta,H))@z[20:35]
        return np.exp(z[2])*r
    def residual(z):
        rel=contacts-z[:2];theta=np.arctan2(rel[:,1],rel[:,0])
        out=(np.linalg.norm(rel,axis=1)-radius(theta,z))/.02
        out=np.r_[out,(z[2]-np.log(4.9))/.1]
        if len(z)>5: out=np.r_[out,z[5:20]/sigma,z[20:35]/sigma]
        return out
    best=None
    for centre in (contacts.mean(axis=0),.5*(contacts.min(axis=0)+contacts.max(axis=0))):
        for a in np.linspace(0,2*np.pi,6,endpoint=False):
            for b in np.linspace(0,2*np.pi,5,endpoint=False):
                fit=least_squares(residual,np.r_[centre,np.log(4.9),a,b],method='trf',max_nfev=300)
                if best is None or fit.cost<best.cost: best=fit
    full=least_squares(residual,np.r_[best.x,np.zeros(30)],method='trf',max_nfev=600)
    z=full.x;rel=contacts-z[:2]
    rms=float(np.sqrt(np.mean((np.linalg.norm(rel,axis=1)-radius(np.arctan2(rel[:,1],rel[:,0]),z))**2)))
    return z,rms,radius
initial=sys.stdin.readline();task=json.loads(initial)
if attack=='over_budget':
    # Repeated valid, safe endpoint requests must consume the authenticated
    # accepted-demonstration budget, even if their trajectories are identical.
    positions=task['initial_demos'][0]['positions']
    for _ in range(task['budget']-len(task['initial_demos'])+1):
        print(json.dumps({'op':'query','start':positions[0],'goal':positions[-1]}),flush=True)
        if not sys.stdin.readline():sys.exit(0)
    raise RuntimeError('evaluator incorrectly allowed too many demonstrations')
elif attack=='family_fit':
    # Writing down the generating process: relay the reference acquisition,
    # then fit (a) the paper's exact g3 polynomial (rotation, scale, centre)
    # and (b) the visible development complex-boundary generator to every
    # tight state, and classify with whichever public family fits better.
    from scipy.optimize import least_squares
    child=subprocess.Popen(RELAY,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    child.stdin.write(initial);child.stdin.flush()
    demos=list(task['initial_demos'])
    while True:
        line=child.stdout.readline();message=json.loads(line)
        print(line.strip(),flush=True)
        reply=sys.stdin.readline()
        if message.get('op')=='finish':
            q=np.asarray(json.loads(reply)['points']);child.kill();break
        if json.loads(reply).get('demo'): demos.append(json.loads(reply)['demo'])
        child.stdin.write(reply);child.stdin.flush()
    contacts=[];normals=[]
    for d in demos:
        pos=np.asarray(d['positions'],float);curv=2*pos[1:-1]-pos[:-2]-pos[2:]
        tight=np.abs(curv).sum(axis=1)>1e-4
        contacts.append(pos[1:-1][tight]);normals.append(-curv[tight]/np.linalg.norm(curv[tight],axis=1,keepdims=True))
    contacts=np.vstack(contacts);normals=np.vstack(normals)
    def g3(points,z):
        theta,log_scale,cx,cy=z;co,si=np.cos(theta),np.sin(theta)
        rotation=np.array([[co,si],[-si,co]]);w=np.exp(log_scale)
        local=(points-[cx,cy])@rotation.T/w;x,y=local.T;a=.25*x*x+y*y-25
        value=(-a*a-50*x*x+687.5)/687.5
        grad=np.column_stack([-a*x-100*x,-4*a*y])@rotation/w/687.5
        return value,grad
    def residuals(z):
        value,grad=g3(contacts,z);norm=np.maximum(np.linalg.norm(grad,axis=1),1e-9)
        return np.r_[value/norm,1-np.sum(grad*normals,axis=1)/norm]
    best=None
    centre=contacts.mean(axis=0)
    for theta in np.linspace(0,np.pi,12,endpoint=False):
        for log_scale in (np.log(.9),np.log(1.)):
            fit=least_squares(residuals,[theta,log_scale,*centre],method='trf',max_nfev=400)
            if best is None or fit.cost<best.cost: best=fit
    rms=float(np.sqrt(np.mean(residuals(best.x)[:len(contacts)]**2)))
    try:
        stress,stress_rms,stress_radius=fit_stress_family(contacts)
    except (ValueError,np.linalg.LinAlgError):
        stress_rms=float('inf')
    print(json.dumps({'g3_fit_rms':rms,'stress_fit_rms':min(stress_rms,1e9)}),file=sys.stderr,flush=True)
    if stress_rms<min(rms,.3):
        rel=q-stress[:2]
        pred=(np.linalg.norm(rel,axis=1)>=stress_radius(np.arctan2(rel[:,1],rel[:,0]),stress)).astype(int)
    elif rms<.3:
        pred=(g3(q,best.x)[0]<=0).astype(int)
    else:
        # Not this public family: polar interpolation of the tight states.
        rel=contacts-centre;angle=np.arctan2(rel[:,1],rel[:,0]);radius=np.linalg.norm(rel,axis=1)
        order=np.argsort(angle);angle=angle[order];radius=radius[order]
        angle=np.r_[angle[-1]-2*np.pi,angle,angle[0]+2*np.pi];radius=np.r_[radius[-1],radius,radius[0]]
        rq=q-centre;qa=np.arctan2(rq[:,1],rq[:,0]);right=np.clip(np.searchsorted(angle,qa),1,len(angle)-1)
        weight=(qa-angle[right-1])/np.maximum(angle[right]-angle[right-1],1e-12)
        pred=(np.linalg.norm(rq,axis=1)>=(1-weight)*radius[right-1]+weight*radius[right]).astype(int)
    print(json.dumps({'predictions':pred.tolist()}),flush=True)
elif attack in ('false_count','negative_count','nan_count','negative_inf_count','move_baseline'):
    child=subprocess.Popen(RELAY,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
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
    # Tight states of the shared initial demonstrations (nonzero curvature).
    initial=np.vstack([np.asarray(d['positions'])[1:-1][np.abs(2*np.asarray(d['positions'])[1:-1]
              -np.asarray(d['positions'])[:-2]-np.asarray(d['positions'])[2:]).sum(axis=1)>1e-6]
              for d in task['initial_demos']])
    print(json.dumps({'op':'finish'}),flush=True)
    q=np.asarray(json.loads(sys.stdin.readline())['points']);n=len(q)
    if attack=='oracle':
        from data.constraints import g_wavy
        # Restore the old public example's coordinate convention. The hidden
        # rotations, geometry and instances remain unavailable to this oracle.
        pred=(g_wavy(q/5)<=0).astype(int)
    elif attack=='contact_disk':
        # Simple learner without acquisition: a disk through the initial
        # contacts (centroid, mean radius).
        if len(initial):
            centre=initial.mean(axis=0);radius=np.linalg.norm(initial-centre,axis=1).mean()
            pred=(np.linalg.norm(q-centre,axis=1)>=radius).astype(int)
        else: pred=np.ones(n,int)
    elif attack=='centered_disk':
        # Prior-shape guess with no demonstrations: unsafe inside a disk of
        # the typical obstacle radius around the workspace origin.
        pred=(np.linalg.norm(q,axis=1)>=5.).astype(int)
    elif attack=='grid_order': pred=(np.arange(n)>=n//2).astype(int)
    elif attack=='grid_resize': pred=np.ones(32,int)
    elif attack=='replace_ground_truth': pred=(q[:,0]<=0).astype(int)
    elif attack=='impossible_predictions': pred=np.full(n,2)
    elif attack=='constant_unsafe': pred=np.zeros(n,int)
    else: pred=np.ones(n,int)
    result={'predictions':pred.tolist()}
    if attack=='empty_predictions': result={'predictions':[]}
    if attack=='null_predictions': result={'predictions':[None]*n}
    if attack=='grid_resize':result['settings']={'eval_grid':8}
    if attack=='forge_flags':result.update(success=True,passed=True,truth=pred.tolist(),accuracy=1)
    print(json.dumps(result),flush=True)
