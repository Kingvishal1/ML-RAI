#!/usr/bin/env python3
"""Train one-ball V20 release network from verified, varied expert demonstrations.

Expert actions are TRAINING LABELS only. Student runtime uses model predictions,
never expert release actions. This is behavioural cloning, not RL or end-to-end ML.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def read_episode(path):
    with np.load(path,allow_pickle=False) as z:
        required=['placement_obs_37','placement_actions_14', 'final_placement_verified',
                  'ground_contact_verified','ball_released_verified','no_virtual_attachment',
                  'final_ball_xy_error_m','final_ball_xy_tolerance_m','placement_label_source','source']
        missing=[k for k in required if k not in z.files]
        if missing:raise ValueError('missing '+', '.join(missing))
        for flag in ['final_placement_verified','ground_contact_verified','ball_released_verified','no_virtual_attachment']:
            if not bool(np.asarray(z[flag]).item()):raise ValueError(flag+' is false')
        if 'SCRIPTED_EXPERT' not in str(z['source']):raise ValueError('unknown teacher provenance')
        if str(z['placement_label_source'])!='EXPERT_JOINT_REFERENCE_AND_RELEASE_DECISION':
            raise ValueError('wrong teacher label contract')
        err=float(z['final_ball_xy_error_m']);tol=float(z['final_ball_xy_tolerance_m'])
        if not np.isfinite(err+tol) or err>tol or tol>0.5 or tol<=0:
            raise ValueError('ball did not land within verified zone')
        x=np.asarray(z['placement_obs_37'],dtype=np.float32)
        y=np.asarray(z['placement_actions_14'],dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=37 or y.shape!=(len(x),14) or len(x)<15:
            raise ValueError(f'bad observation/action arrays {x.shape} {y.shape}')
        if not (np.isfinite(x).all() and np.isfinite(y).all()):raise ValueError('nonfinite data')
        if not np.isin(y[:,13],[0,1]).all() or np.count_nonzero(y[:,13]==1)<5 or np.count_nonzero(y[:,13]==0)<10:
            raise ValueError('insufficient actual hold/release supervision')
        if np.any(y[:,7:13]<-.01) or np.any(y[:,7:13]>2.01):
            raise ValueError('invalid finger commands')
        if np.any(np.abs(y[:,:7])>.17):raise ValueError('unphysical joint increments')
        digest=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        return x,y,digest,err


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demos',default='v20_release_demos')
    ap.add_argument('--out',default='trained_v20_one_ball_release')
    ap.add_argument('--epochs',type=int,default=140)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    collected=[];seen=set();audit=[]
    for f in sorted(Path(args.demos).glob('expert_*.npz')):
        try:
            x,y,digest,err=read_episode(f)
            if digest in seen:raise ValueError('duplicate observations and actions')
            seen.add(digest); collected.append((f.name,x,y))
            audit.append({'file':f.name,'accepted':True,'frames':len(x),'landing_error_m':err})
        except (ValueError,KeyError, OSError) as exc:
            audit.append({'file':f.name,'accepted':False,'reason':str(exc)})
    if len(collected)<4:
        print(json.dumps(audit,indent=2))
        raise SystemExit(f'NEED FOUR DISTINCT SUCCESSFUL ONE-BALL DEMOS. Found {len(collected)}; no weights created.')
    gen=np.random.default_rng(args.seed)
    order=gen.permutation(len(collected))
    val=[collected[int(order[0])]]
    train=[collected[int(i)] for i in order[1:]]
    X=np.concatenate([e[1] for e in train]); Y=np.concatenate([e[2] for e in train]);
    XV=np.concatenate([e[1] for e in val]);YV=np.concatenate([e[2] for e in val]);
    mean=X.mean(0); std=np.maximum(X.std(0),np.array([.05]*37,dtype=np.float32))
    def norm(x):return torch.as_tensor(np.clip((x-mean)/std,-10,10),dtype=torch.float32)
    xt,yt=norm(X),torch.as_tensor(Y);xv,yv=norm(XV),torch.as_tensor(YV)
    model=torch.nn.Sequential(torch.nn.Linear(37,128),torch.nn.Tanh(),
                torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,14))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    # Positive release frames are rare; avoid a model that only predicts HOLD.
    p=max(1,int(np.count_nonzero(Y[:,13]>=.5)))
    n=max(1,int(np.count_nonzero(Y[:,13]<.5)))
    pos_weight=torch.tensor(min(40.0,n/p),dtype=torch.float32)
    weights=torch.where(yt[:,13]>.5,torch.tensor(min(15., n/p)),torch.tensor(1.))
    sampler=torch.utils.data.WeightedRandomSampler(weights, max(len(weights),512), replacement=True)
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def loss(z,y):
        # 13 learned motor commands + learned release event classification.
        physical=torch.mean((z[:,:7]-y[:,:7])**2)/.02 + torch.mean((z[:,7:13]-y[:,7:13])**2)
        return physical + 0.30*bce(z[:,13],y[:,13])
    best=float('inf');best_state=None;best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train()
        for ids in torch.utils.data.DataLoader(list(sampler),batch_size=128):
            yp=model(xt[ids]);lv=loss(yp,yt[ids]); opt.zero_grad();lv.backward();opt.step()
        model.eval()
        with torch.no_grad():
            yp=model(xv);score=float(loss(yp,yv))
        if score<best:
            best=score;best_epoch=epoch
            best_state=[p.detach().cpu().numpy().copy() for p in model.parameters()]
        if epoch%20==0 or epoch==1:
            print(f'EPOCH {epoch:03d} validation_loss={score:.5f} best={best:.5f}',flush=True)
    with torch.no_grad():
        for param,value in zip(model.parameters(),best_state):param.copy_(torch.from_numpy(value))
        outv=model(xv); probs=torch.sigmoid(outv[:,13]).numpy()
        pred=(probs>=.5);truth=YV[:,13]>=.5
        recall=float(np.sum(pred & truth)/max(1,np.sum(truth)))
        precision=float(np.sum(pred & truth)/max(1,np.sum(pred)))
        false_positive_rate=float(np.sum(pred & ~truth)/max(1,np.sum(~truth)))
    out=Path(args.out)
    if out.exists():raise SystemExit('Output directory exists: '+str(out)+' (no overwrite)')
    out.mkdir(parents=True)
    linear=[m for m in model if isinstance(m,torch.nn.Linear)]
    data={'format':np.asarray('v20_neural_release_bc_v1'),'skill':np.asarray('placement'),
          'trained':np.asarray(True),'teacher_source':np.asarray('SCRIPTED_EXPERT_DEMONSTRATIONS'),
          'layers':np.asarray(len(linear)),'xmean':mean.astype(np.float64),'xstd':std.astype(np.float64)}
    for i,layer in enumerate(linear):
        data[f'w{i}']=layer.weight.detach().numpy().T.astype(np.float64)
        data[f'b{i}']=layer.bias.detach().numpy().astype(np.float64)
    np.savez_compressed(out/'release_student.npz',**data)
    report={'source':'BC imitation of original V20 scripted expert (teacher only)',
            'student_release_scripted_at_runtime':False,
            'remaining_scripted_at_runtime':['carry navigation','physical gating and phase progression'],
            'train_episodes':[e[0] for e in train],'heldout_episode':val[0][0],
            'best_epoch':best_epoch,'validation_loss':best,
            'heldout_release_recall':recall,'heldout_release_precision':precision,
            'heldout_release_false_positive_rate':false_positive_rate,
            'training_frames':len(X),'validation_frames':len(XV),'closed_loop_verified':False,
            'audit':audit}
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TRAINED release_student.npz',out,'release_recall',recall,'precision',precision,
          'false_positive_rate',false_positive_rate,flush=True)
    print('NOT YET closed-loop verified. RUN_V20_TEST_ONE_BALL_RELEASE.sh is required.')

if __name__=='__main__':main()
