#!/usr/bin/env python3
"""Train neural return-home action by BC (not reinforcement learning).

Uses a labeled reference-target oracle only during training; NEVER at inference.
Optional real simulator expert traces are included if provided. Result is a
fresh learned model; successful physics return still requires a rollout test.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='trained_v21_return_home_bc/return_home_bc.npz')
    ap.add_argument('--expert_npz', action='append', default=[])
    ap.add_argument('--samples', type=int, default=36000)
    ap.add_argument('--epochs', type=int, default=85)
    ap.add_argument('--seed', type=int, default=20261105)
    args = ap.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    # Broad joint-error data and a concentrated near-home subset.
    n = args.samples
    errors = rng.uniform(-3.5, 3.5, size=(n,7))
    errors[:n//2] = rng.normal(0, 0.30, size=(n//2,7))
    errors[:n//5] = rng.normal(0, 0.055, size=(n//5,7))
    velocities = rng.uniform(-0.25,0.25,size=(n,7))
    velocities[:n//5] *= .25
    # Training-only teacher maps state to a bounded next joint reference.
    obs = np.concatenate([errors, velocities],axis=1).astype(np.float32)
    actions = np.clip(errors - 0.35 * velocities, -.15, .15).astype(np.float32)
    for filename in args.expert_npz:
        with np.load(filename, allow_pickle=False) as d:
            if str(d['source']) != 'EXPERT_RETURN_HOME_SERVO':
                raise ValueError(f'{filename}: not labelled expert data')
            if not bool(d['successful']):
                raise ValueError(f'{filename}: demonstration not verified successful')
            additional_obs = np.asarray(d['obs'], dtype=np.float32)
            additional_actions = np.asarray(d['action_delta'], dtype=np.float32)
        if additional_obs.ndim != 2 or additional_obs.shape[1] != 14 or additional_actions.shape != (len(additional_obs),7):
            raise ValueError(f'{filename}: wrong data dimensions')
        # Weight real demonstrated states 10x without changing labels.
        obs = np.concatenate([obs, np.repeat(additional_obs, 10, axis=0)])
        actions = np.concatenate([actions, np.repeat(additional_actions, 10, axis=0)])
    order = rng.permutation(len(obs))
    train_ids, val_ids = order[:int(.9*len(order))], order[int(.9*len(order)):]
    mean = obs[train_ids].mean(axis=0)
    std = np.maximum(obs[train_ids].std(axis=0), .02)
    x = torch.from_numpy((obs-mean)/std)
    y = torch.from_numpy(actions)
    model = torch.nn.Sequential(torch.nn.Linear(14,64),torch.nn.Tanh(),
                                torch.nn.Linear(64,64),torch.nn.Tanh(),
                                torch.nn.Linear(64,7))
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-6)
    best_val = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        shuffled = torch.randperm(len(train_ids))
        for inds in shuffled.split(512):
            pred = model(x[train_ids[inds]])
            loss = torch.nn.functional.mse_loss(pred,y[train_ids[inds]])
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            val = torch.nn.functional.mse_loss(model(x[val_ids]),y[val_ids]).item()
        if val < best_val:
            best_val = val
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0 or epoch==args.epochs-1:
            print(f'EPOCH {epoch:03d} val_mse_rad2={val:.8f}',flush=True)
    model.load_state_dict(best_state)
    params = {}
    for i,idx in enumerate((0,2,4)):
        layer=model[idx]
        params[f'w{i}']=layer.weight.detach().cpu().numpy()
        params[f'b{i}']=layer.bias.detach().cpu().numpy()
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:
        np.savez_compressed(f, format=np.asarray('robotnik_return_home_bc_v1'),
            num_layers=np.asarray(3),max_step=np.asarray(.15),
            obs_mean=mean, obs_std=std, val_mse=np.asarray(best_val),
            **params)
    print(f'RETURN_HOME_BC_SAVED path={dest} val_mse_rad2={best_val:.8f} samples={len(obs)}')
    # Offline sanity check is not proof of a physical trajectory.
    from importlib.util import spec_from_file_location, module_from_spec
    spec=spec_from_file_location('learned_return_home_policy',str(Path(__file__).resolve().parents[1]/'robotnik_sensor_bc_v2'/'learned_return_home_policy.py'))
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    policy=mod.LearnedReturnHomePolicy.load(dest)
    print('NEURAL_ONLY_SANITY max_delta=',np.max(np.abs(policy.predict_delta(np.ones(7),np.zeros(7),np.zeros(7)))))

if __name__=='__main__': main()
