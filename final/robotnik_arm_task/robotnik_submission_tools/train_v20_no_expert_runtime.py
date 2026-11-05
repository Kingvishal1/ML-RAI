#!/usr/bin/env python3
"""Train fully learned V20 carry and placement policies from VERIFIED demonstrations.

At runtime, no expert steering, arm hold, lowering, or opening actions are used.
The training data source restriction is HUMAN TELEOP only. Expert-generated
trajectory labels are not accepted, even if they were successful in simulation.
No demonstration -> no weights -> no fictional 'working trained model'.
"""
import argparse
from pathlib import Path
import sys, json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'robotnik_sensor_bc_v2'))
from robotnik_learned_task_policy import FORMAT, DIMS

MIN_N=4
FLOORS={
 'carry_full': np.r_[[.05,.05,.12,.08,.2],[.10]*7,[.50]*7,[.03]*3,[.10]*3],
 'placement_full': np.r_[[.10]*7,[.25]*7,[.10]*6,[.50]*6,[.05]*3,[.10]*3,[.03]*3,[.05,1.0]],
}
CONTRACTS={'carry_full':('carry_full_obs_25','carry_full_actions_9'),
           'placement_full':('placement_obs_37','placement_actions_14')}

def valid_episode(path):
    with np.load(path, allow_pickle=False) as z:
        keys=set(z.files)
        require=['final_placement_verified','ground_contact_verified',
                 'ball_released_verified','no_virtual_attachment',
                 'final_ball_xy_error_m','drop_zone_radius_m','source',
                 'carry_full_obs_25','carry_full_actions_9',
                 'placement_obs_37','placement_actions_14']
        missing=[k for k in require if k not in keys]
        if missing: raise ValueError('Missing verified sensor/label arrays: '+','.join(missing))
        source=str(z['source']).upper()
        if 'TELEOP' not in source or 'SCRIPTED_EXPERT' in source:
            raise ValueError('No permitted demonstration provenance. Default requires human TELEOP data.')
        for name in require[:4]:
            if not bool(np.asarray(z[name]).item()):
                raise ValueError(f'{name} is false')
        d=float(np.asarray(z['final_ball_xy_error_m']).item())
        r=float(np.asarray(z['drop_zone_radius_m']).item())
        if not (np.isfinite(d) and np.isfinite(r) and .15<=r<=1.5 and d<=r-.03):
            raise ValueError('Ball not verified fully inside circle')
        result={}
        for skill,(xk,yk) in CONTRACTS.items():
            x=np.asarray(z[xk],dtype=np.float32)
            y=np.asarray(z[yk],dtype=np.float32)
            nx,ny=DIMS[skill]
            if x.ndim!=2 or x.shape[1]!=nx or y.shape!=(len(x),ny):
                raise ValueError(f'Invalid {skill} shapes: {x.shape} {y.shape}')
            if len(x)<(40 if skill=='carry_full' else 30):
                raise ValueError(f'Not enough {skill} samples')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite observations/actions')
            if skill=='carry_full':
                if np.any((y[:,0]<-1e-4)|(y[:,0]>.101)) or np.any(np.abs(y[:,1])>1.01):
                    raise ValueError('Unphysical carry wheel action labels')
                if np.any(np.abs(y[:,2:])>40.01):
                    raise ValueError('Unphysical arm torque labels')
            else:
                if np.any(np.abs(y[:,:7])>.16) or np.any(y[:,7:13]<-.001) or np.any(y[:,7:13]>2.01):
                    raise ValueError('Unphysical placement action labels')
                if np.any(~np.isin(y[:,13],[0,1])):
                    raise ValueError('Placement release labels must be 0 or 1')
                if np.count_nonzero(y[:,13]==1)<5 or np.count_nonzero(y[:,13]==0)<10:
                    raise ValueError('Need meaningful grasp/placement AND actual opening labels')
            result[skill]=(x.copy(),y.copy())
        digest=hashlib.sha256(b''.join(a.tobytes() for pair in result.values() for a in pair)).hexdigest()
        return result,source,digest

def train_one(skill, train_eps, val_eps, epochs,seed,device):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    x=np.concatenate([e[skill][0] for e in train_eps])
    y=np.concatenate([e[skill][1] for e in train_eps])
    xv=np.concatenate([e[skill][0] for e in val_eps])
    yv=np.concatenate([e[skill][1] for e in val_eps])
    xmean=x.mean(axis=0).astype(np.float64)
    xscale=np.maximum(x.std(axis=0),FLOORS[skill]).astype(np.float64)
    ymean=y.mean(axis=0).astype(np.float64)
    yscale=np.maximum(y.std(axis=0),0.08).astype(np.float64)
    if skill=='placement_full':
        ymean[-1]=0.0; yscale[-1]=1.0
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],96),torch.nn.Tanh(),
        torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,y.shape[1])).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=0.0005,weight_decay=0.0001)
    X=torch.as_tensor((x-xmean)/xscale,device=device,dtype=torch.float32)
    Y=torch.as_tensor((y-ymean)/yscale,device=device,dtype=torch.float32)
    XV=torch.as_tensor((xv-xmean)/xscale,device=device,dtype=torch.float32)
    YV=torch.as_tensor((yv-ymean)/yscale,device=device,dtype=torch.float32)
    pos=float(np.sum(y[:,13]==1)) if skill=='placement_full' else 0
    neg=float(np.sum(y[:,13]==0)) if skill=='placement_full' else 0
    posw=torch.tensor(min(50.0,max(1.0,neg/max(1.0,pos))),device=device)
    mse=torch.nn.MSELoss()
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    best=float('inf'); best_weights=None; best_epoch=0; impatience=0
    def loss(output,labels):
        if skill=='carry_full': return mse(output,labels)
        return mse(output[:,:13],labels[:,:13]) + 0.35*bce(output[:,13],labels[:,13])
    for epoch in range(1,epochs+1):
        net.train()
        for ids in torch.randperm(len(X),device=device).split(256):
            prediction=net(X[ids]); value=loss(prediction,Y[ids])
            if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
            opt.zero_grad();value.backward();opt.step()
        net.eval()
        with torch.no_grad():score=float(loss(net(XV),YV).item())
        if score<best-1e-7:
            best=score;best_epoch=epoch;impatience=0
            best_weights=[(m.weight.detach().cpu().numpy().T.copy().astype(np.float64),
                           m.bias.detach().cpu().numpy().copy().astype(np.float64))
                for m in net if isinstance(m,torch.nn.Linear)]
        else:impatience+=1
        if impatience>=45:break
    with torch.no_grad():
        h=(xv-xmean)/xscale
        for i,(w,b) in enumerate(best_weights):
            h=h@w+b
            if i<len(best_weights)-1:h=np.tanh(h)
        pred=h*yscale+ymean
        if skill=='placement_full':pred[:,13]=1/(1+np.exp(-np.clip(h[:,13],-50,50)))
    # Runtime must not clip to the min/max observed in a finite small dataset.
    # Use defensible mechanical command limits instead.
    if skill=='carry_full':
        amin=np.r_[[0,-1],[-40]*4,[-20]*3]
        amax=np.r_[[.1,1],[40]*4,[20]*3]
    else:
        amin=np.r_[[-.15]*7,[0]*6,[0]]
        amax=np.r_[[.15]*7,[2]*6,[1]]
    arrays={'format':np.asarray(FORMAT),'skill':np.asarray(skill),
       'training_episodes':np.asarray(len(train_eps)),'trained':np.asarray(True),
       'layers':np.asarray(3),'x_mean':xmean,'x_scale':xscale,
       'y_mean':ymean,'y_scale':yscale,'action_min':amin.astype(float),
       'action_max':amax.astype(float)}
    for i,(w,b) in enumerate(best_weights):arrays[f'w{i}']=w;arrays[f'b{i}']=b
    report={'train_frames':len(x),'val_frames':len(xv),'best_epoch':best_epoch,
      'heldout_loss':best,'heldout_mae_by_action':np.abs(pred-yv).mean(axis=0).tolist(),
      'closed_loop_verified':False}
    if skill=='placement_full':
        report['heldout_release_accuracy']=float(np.mean((pred[:,13]>=.5)==(yv[:,13]>=.5)))
        report['heldout_release_positive_fraction']=float(np.mean(yv[:,13]>=.5))
    return arrays,report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demos',required=True,help='Folder containing distinct verified TELEOP .npz episodes')
    ap.add_argument('--output',required=True,help='New empty output folder; no overwrites')
    ap.add_argument('--epochs',type=int,default=300)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    if a.epochs<1:ap.error('epochs must be positive')
    output=Path(a.output)
    if output.exists():ap.error('Output already exists; select a new directory')
    files=sorted(Path(a.demos).glob('*.npz'))
    accepted=[];hashes=set();audit=[]
    for f in files:
        try:
            ep,source,h=valid_episode(f)
            if h in hashes:raise ValueError('Identical duplicate episode')
            hashes.add(h);accepted.append((f.name,ep,source))
            audit.append({'file':f.name,'accepted':True,'source':source,'hash':h})
        except (ValueError,KeyError,OSError) as exc:
            audit.append({'file':f.name,'accepted':False,'reason':str(exc)})
    if len(accepted)<MIN_N+1:
        for entry in audit:print(entry)
        raise SystemExit(f'NO MODEL TRAINED: need >=5 distinct verified episodes '
            f'(>=4 train, >=1 entire-episode validation), got {len(accepted)}')
    rng=np.random.default_rng(a.seed)
    ix=rng.permutation(len(accepted));nval=max(1,int(np.ceil(len(ix)*.2)))
    train=[accepted[int(i)][1] for i in ix[nval:]]
    val=[accepted[int(i)][1] for i in ix[:nval]]
    if len(train)<MIN_N:raise SystemExit('Not enough training episodes after held-out split')
    models={};report={'validation_episodes':[accepted[int(i)][0] for i in ix[:nval]],
      'training_episodes':[accepted[int(i)][0] for i in ix[nval:]],
      'source_audit':audit,'closed_loop_verified':False,'skills':{}}
    for skill in CONTRACTS:
        models[skill],report['skills'][skill]=train_one(skill,train,val,a.epochs,a.seed,a.device)
    output.mkdir(parents=True)
    for skill,params in models.items():
        with (output/f'{skill}.npz').open('xb') as fp:np.savez_compressed(fp,**params)
    (output/'train_report.json').write_text(json.dumps(report,indent=2))
    print('TRAINED MODEL CANDIDATES:',output)
    print('Teacher data provenance:',sorted(set(v[2] for v in accepted)))
    print('NOT closed-loop validated. Never claim 2/2 from imitation loss alone.')

if __name__=='__main__':main()
