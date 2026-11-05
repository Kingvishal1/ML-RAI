#!/usr/bin/env python3
"""Offline behavioural cloning from VERIFIED Robotnik expert carry/release NPZ traces.

Trains two *new* candidate skills: 5->2 navigation (v,w) and 17->6 release
joint targets. Does not change existing trained pickup/base/arm/grasp models.
Does not train an arrival trigger; release policy applies ONLY after arrival.
Requires >=2 independent, genuinely successful simulator episodes. This
script cannot run the Robotnik simulator or validate learned closed-loop control.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

MODEL_FORMAT = 'robotnik_expert_bc_mlp_v1'
SPECS = {
    'carry': ('obs', 'actions_vw', 5, 2, np.array([.05,.05,.12,.08,.2])),
    'release': ('release_obs_17', 'release_expert_finger_targets_rad', 17, 6,
                np.array([.03]*3+[.10]*6+[.50]*6+[.05]*2)),
}


def get_verified(path):
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        need = {'physical_success_verified','ball_released_verified','no_virtual_attachment'}
        if not need.issubset(keys):
            raise ValueError(f'{path}: not a V8.2 expert trace; missing {need-keys}')
        good = all(bool(z[x].item()) for x in need)
        if not good:
            print(f'SKIP unverified/failed episode: {path}', flush=True)
            return None
        if 'release_label_source' not in keys or 'EXPLICIT_PROGRAMMED_EXPERT_OPEN' not in str(z['release_label_source']):
            raise ValueError(f'{path}: no verified expert release action labels')
        if 'source' not in keys or 'SCRIPTED_EXPERT' not in str(z['source']):
            raise ValueError(f'{path}: carry actions are not known expert labels')
        episodes = {}
        for skill, (xkey,ykey,nx,ny,_) in SPECS.items():
            if xkey not in keys or ykey not in keys:
                raise ValueError(f'{path}: missing {xkey}/{ykey}')
            x = np.asarray(z[xkey], dtype=np.float32)
            y = np.asarray(z[ykey], dtype=np.float32)
            if x.ndim != 2 or y.ndim != 2 or x.shape[1] != nx or y.shape != (len(x),ny):
                raise ValueError(f'{path}: invalid {skill} shapes: x={x.shape}, y={y.shape}')
            if len(x) < (30 if skill=='carry' else 5):
                raise ValueError(f'{path}: too few {skill} samples ({len(x)})')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'{path}: nonfinite samples in {skill}')
            if skill == 'carry' and (np.any(np.abs(y[:,0]) > .11) or np.any(np.abs(y[:,1]) > 1.01)):
                raise ValueError(f'{path}: carry actions outside documented physical bounds')
            if skill == 'release' and (np.any(y < -.001) or np.any(y > 2.01)):
                raise ValueError(f'{path}: release joint targets outside physical bounds')
            episodes[skill] = (x.copy(),y.copy())
        return episodes


def train_skill(skill, train_episodes, val_episodes, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    x = np.concatenate([e[skill][0] for e in train_episodes],axis=0)
    y = np.concatenate([e[skill][1] for e in train_episodes],axis=0)
    xv = np.concatenate([e[skill][0] for e in val_episodes],axis=0)
    yv = np.concatenate([e[skill][1] for e in val_episodes],axis=0)
    xmin = x.min(axis=0); xmax = x.max(axis=0)
    xmean = x.mean(axis=0).astype(np.float64)
    xscale = np.maximum(x.std(axis=0), SPECS[skill][4]).astype(np.float64)
    ymean = y.mean(axis=0).astype(np.float64)
    yscale = np.maximum(y.std(axis=0), .03 if skill=='carry' else .10).astype(np.float64)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.Tanh(),
                                torch.nn.Linear(64,64),torch.nn.Tanh(),
                                torch.nn.Linear(64,y.shape[1])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.0001)
    xx = torch.as_tensor((x-xmean)/xscale,dtype=torch.float32,device=device)
    yy = torch.as_tensor((y-ymean)/yscale,dtype=torch.float32,device=device)
    vv = torch.as_tensor((xv-xmean)/xscale,dtype=torch.float32,device=device)
    best_loss = float('inf'); best_weights=None; best_epoch=0; patience=65; stalled=0
    for epoch in range(epochs):
        model.train()
        perm=torch.randperm(len(xx),device=device)
        for ix in perm.split(256):
            pred=model(xx[ix]); loss=torch.mean((pred-yy[ix])**2)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite {skill} training loss')
            optim.zero_grad();loss.backward();optim.step()
        model.eval()
        with torch.no_grad():
            pred=model(vv).cpu().numpy()*yscale+ymean
            loss=float(np.mean((pred-yv)**2))
        if loss < best_loss-1e-9:
            best_loss=loss;best_epoch=epoch+1;stalled=0
            best_weights=[(mod.weight.detach().cpu().numpy().T.astype(np.float64).copy(),
                           mod.bias.detach().cpu().numpy().astype(np.float64).copy())
                          for mod in model if isinstance(mod,torch.nn.Linear)]
        else:
            stalled += 1
        if stalled >= patience:
            break
    # Pure-NumPy reference inference from best checkpoint (no Torch required at inference).
    h=(xv-xmean)/xscale
    for i,(w,b) in enumerate(best_weights):
        h=h@w+b
        if i != len(best_weights)-1:
            h=np.tanh(h)
    pred=h*yscale+ymean
    # Show direct imitation error AND a constant baseline; don't imply closed-loop success.
    mean_baseline=np.broadcast_to(ymean, yv.shape)
    report={
        'skill':skill,'train_frames':int(len(x)),'validation_frames':int(len(xv)),
        'validation_mse':float(np.mean((pred-yv)**2)),
        'validation_mae_per_action':np.abs(pred-yv).mean(axis=0).tolist(),
        'constant_baseline_mae_per_action':np.abs(mean_baseline-yv).mean(axis=0).tolist(),
        'best_epoch':best_epoch,'release_targets_label_variation_rad':float(np.max(np.std(y,axis=0))) if skill=='release' else None,
        'physical_closed_loop_validated':False,
        'warning':('Expert open targets can be constant: a learned release action is not a learned arrival decision. '
                   'Never use this output to grasp or to decide when to open.') if skill=='release' else
                  'Expert carry imitation is not closed-loop navigation validation.',
    }
    contents={'format':np.asarray(MODEL_FORMAT),'skill':np.asarray(skill),'layers':np.asarray(len(best_weights)),
              'x_mean':xmean,'x_scale':xscale,'y_mean':ymean,'y_scale':yscale,
              'action_min':y.min(axis=0).astype(np.float64),'action_max':y.max(axis=0).astype(np.float64),
              'observed_x_min':xmin.astype(np.float64),'observed_x_max':xmax.astype(np.float64)}
    for i,(w,b) in enumerate(best_weights):
        contents[f'w{i}']=w;contents[f'b{i}']=b
    return contents, report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True,help='V8.2 --record_expert_npz outputs (not logs)')
    p.add_argument('--out',required=True,help='NEW output directory for candidate carry/release models')
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    paths=list(dict.fromkeys(str(Path(x).resolve()) for x in a.data))
    if len(paths)<2:p.error('Need at least 2 different successful expert demonstrations; 5+ recommended')
    episodes=[];good_paths=[];failed_paths=[]
    for path in paths:
        ep=get_verified(path)
        if ep is None:failed_paths.append(Path(path).name)
        else:episodes.append(ep);good_paths.append(Path(path).name)
    if len(episodes)<2:
        p.error(f'Only {len(episodes)} verified successes. Collect >=2; failed runs were skipped.')
    rng=np.random.default_rng(a.seed);perm=rng.permutation(len(episodes));n_val=max(1,round(len(episodes)*.2))
    val_eps=[episodes[i] for i in perm[:n_val]]
    train_eps=[episodes[i] for i in perm[n_val:]]
    output=Path(a.out)
    if output.exists():p.error(f'Output exists; refusing to overwrite: {output}')
    output.mkdir(parents=True)
    report={'source':'verified scripted expert demonstrations; not end-to-end expert',
            'existing_ml_models_unchanged':True,'successful_episode_count':len(episodes),
            'train_episode_count':len(train_eps),'heldout_episode_count':len(val_eps),
            'all_episode_names':good_paths,'failed_episodes_skipped':failed_paths,
            'split':'Independent expert episode holdout; any repeated starts remain correlated',
            'seed':a.seed,'epochs_max':a.epochs,'skills':{}}
    for skill in ('carry','release'):
        data,skill_report=train_skill(skill,train_eps,val_eps,a.epochs,a.seed,a.device)
        file=output/f'{skill}_candidate_bc.npz'
        with file.open('xb') as f:np.savez_compressed(f,**data)
        report['skills'][skill]=skill_report
        print(f'{skill}: validation_mae={skill_report["validation_mae_per_action"]}, '
              f'constant_baseline_mae={skill_report["constant_baseline_mae_per_action"]}',flush=True)
    (output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'CANDIDATE MODELS SAVED: {output}  (NOT physically validated / NOT integrated into expert controller)')

if __name__=='__main__':
    main()
