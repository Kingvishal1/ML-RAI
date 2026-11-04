#!/usr/bin/env python3
"""Train candidate imitation models from verified Robotnik ground-placement traces.

Accepts only real demonstrations where the ball was lowered, detached, made
stable ground contact, and remained within the final XY placement tolerance.
Existing base, arm and V4 grasp neural weights are NEVER overwritten.

Outputs candidate carry (5->2) and release (17->6) BC MLPs, plus audit report.
This is supervised imitation of a HYBRID demonstration, not an end-to-end
pick-and-place policy and not a closed-loop validation of candidate weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FORMAT = 'robotnik_expert_bc_mlp_v1'
SPECS = {
    'carry': ('obs', 'actions_vw', 5, 2, np.array([.05,.05,.12,.08,.2])),
    'release': ('release_obs_17', 'release_expert_finger_targets_rad', 17, 6,
                np.array([.03]*3+[.10]*6+[.50]*6+[.05]*2)),
}


def _flag(z, key):
    return key in z and bool(np.asarray(z[key]).item())


def read_episode(path, arrival_max_error=0.075):
    """Use physical ground-contact and final-position evidence for acceptance."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {'ball_released_verified', 'no_virtual_attachment',
                    'source', 'release_label_source', 'dropoff_xy',
                    'release_ball_goal_error_m', 'release_elapsed_s',
                    'ground_contact_verified', 'final_ball_xy_error_m'}
        if not required.issubset(keys):
            raise ValueError(f'{path}: missing expert evidence: {required - keys}')
        if 'SCRIPTED_EXPERT' not in str(z['source']):
            raise ValueError(f'{path}: navigation labels are not verified expert actions')
        if 'EXPLICIT_PROGRAMMED_EXPERT_OPEN' not in str(z['release_label_source']):
            raise ValueError(f'{path}: no explicit expert opening labels')
        no_attachment = _flag(z, 'no_virtual_attachment')
        released = _flag(z, 'ball_released_verified')
        ground_contact = _flag(z, 'ground_contact_verified')
        final_xy_error = float(np.asarray(z['final_ball_xy_error_m']).item())
        if not ground_contact:
            raise ValueError(f'{path}: stable ball-to-ground contact was not verified')
        if not np.isfinite(final_xy_error) or final_xy_error > 0.05:
            raise ValueError(f'{path}: final ball XY error is {final_xy_error:.3f}m; maximum is 0.05m')
        if 'carries_ball_with_friction' in keys and not _flag(z, 'carries_ball_with_friction'):
            raise ValueError(f'{path}: frictional physical carrying not confirmed')
        target = np.asarray(z['dropoff_xy'], dtype=float)
        release_xy_errors = np.asarray(z['release_ball_goal_error_m'], dtype=float)
        release_times = np.asarray(z['release_elapsed_s'], dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError(f'{path}: invalid target XY')
        if (release_xy_errors.ndim != 1 or len(release_xy_errors) < 5 or
                not np.isfinite(release_xy_errors).all() or
                release_times.shape != release_xy_errors.shape or
                not np.isfinite(release_times).all()):
            raise ValueError(f'{path}: invalid release measurements')
        # The release trace starts after the arm reaches the guarded ground
        # placement pose. Check its first sample, not a later point after fall.
        arrived_by_record = (0 <= release_xy_errors[0] <= arrival_max_error and
                             0 <= release_times[0] <= 0.10)
        explicit_arrival = _flag(z, 'release_at_goal_verified')
        # Even explicit metadata is cross-checked against sensor observations.
        valid = bool(no_attachment and released and arrived_by_record)
        if not valid:
            print(f'SKIP not verified as release at target: {path.name} '
                  f'(no_attachment={no_attachment}, detached_fell={released}, '
                  f'release_start_xy_error={release_xy_errors[0]:.3f}m, '
                  f'release_start_t={release_times[0]:.3f}s)', flush=True)
            return None
        if ('release_at_goal_verified' in keys and not explicit_arrival):
            print(f'SKIP explicit release_at_goal_verified=False: {path.name}', flush=True)
            return None
        episodes = {}
        for skill, (xkey, ykey, nx, ny, _) in SPECS.items():
            if xkey not in keys or ykey not in keys:
                raise ValueError(f'{path}: missing {skill} observations or actions')
            x = np.asarray(z[xkey], dtype=np.float32)
            y = np.asarray(z[ykey], dtype=np.float32)
            if x.ndim != 2 or y.ndim != 2 or x.shape[1] != nx or y.shape != (len(x), ny):
                raise ValueError(f'{path}: invalid {skill} shape: obs={x.shape}, action={y.shape}')
            if len(x) < (30 if skill == 'carry' else 5):
                raise ValueError(f'{path}: insufficient {skill} frames')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'{path}: nonfinite observations/actions')
            if skill == 'carry' and (np.any(np.abs(y[:,0]) > .101) or np.any(np.abs(y[:,1]) > 1.01)):
                raise ValueError(f'{path}: expert navigation actions exceed bounds')
            if skill == 'release' and (np.any(y < -.001) or np.any(y > 2.01)):
                raise ValueError(f'{path}: expert gripper actions exceed joint limits')
            episodes[skill] = (x.copy(), y.copy())
        metadata = {
            'filename': path.name,
            'original_strict_placement_flag': _flag(z, 'physical_success_verified'),
            'explicit_at_target_flag': explicit_arrival if 'release_at_goal_verified' in keys else None,
            'release_at_target_reverified': True,
            'release_start_xy_error_m': float(release_xy_errors[0]),
            'release_end_xy_error_m': float(release_xy_errors[-1]),
            'ground_contact_verified': ground_contact,
            'final_ball_xy_error_m': final_xy_error,
            'target_xy': target.tolist(),
            'carry_frames': len(episodes['carry'][0]),
            'release_frames': len(episodes['release'][0]),
        }
        return episodes, metadata


def train_skill(skill, train_eps, val_eps, epochs, seed, device):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install PyTorch into your comp0245 environment to train') from exc
    torch.manual_seed(seed)
    x = np.concatenate([e[skill][0] for e in train_eps], axis=0)
    y = np.concatenate([e[skill][1] for e in train_eps], axis=0)
    xv = np.concatenate([e[skill][0] for e in val_eps], axis=0)
    yv = np.concatenate([e[skill][1] for e in val_eps], axis=0)
    xmin, xmax = x.min(0), x.max(0)
    xmean = x.mean(0).astype(np.float64)
    xscale = np.maximum(x.std(0), SPECS[skill][4]).astype(np.float64)
    ymean = y.mean(0).astype(np.float64)
    yscale = np.maximum(y.std(0), .03 if skill == 'carry' else .10).astype(np.float64)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],64), torch.nn.Tanh(),
                                torch.nn.Linear(64,64), torch.nn.Tanh(),
                                torch.nn.Linear(64,y.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.0001)
    xx = torch.as_tensor((x-xmean)/xscale, dtype=torch.float32, device=device)
    yy = torch.as_tensor((y-ymean)/yscale, dtype=torch.float32, device=device)
    vv = torch.as_tensor((xv-xmean)/xscale, dtype=torch.float32, device=device)
    best = float('inf'); weights = None; best_epoch = 0; no_improve = 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(xx),device=device).split(256):
            loss = torch.mean((model(xx[idx])-yy[idx])**2)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(vv).cpu().numpy()*yscale + ymean
            val_loss = float(np.mean((pred-yv)**2))
        if val_loss < best - 1e-9:
            best = val_loss; best_epoch = epoch+1; no_improve = 0
            weights = [(m.weight.detach().cpu().numpy().T.astype(np.float64).copy(),
                        m.bias.detach().cpu().numpy().astype(np.float64).copy())
                       for m in model if isinstance(m, torch.nn.Linear)]
        else:
            no_improve += 1
        if no_improve >= 65: break
    # Check the exact saved NumPy inference parameter layout.
    h = (xv-xmean)/xscale
    for i,(w,b) in enumerate(weights):
        h = h @ w + b
        if i < len(weights)-1: h = np.tanh(h)
    pred = h*yscale + ymean
    report = {
        'skill': skill, 'train_frames': len(x), 'heldout_frames': len(xv),
        'heldout_mse': float(np.mean((pred-yv)**2)),
        'heldout_mae_per_action': np.abs(pred-yv).mean(0).tolist(),
        'constant_baseline_mae_per_action': np.abs(yv-ymean).mean(0).tolist(),
        'best_epoch': best_epoch,
        'release_targets_may_be_constant': bool(skill == 'release' and np.max(np.std(y,axis=0)) < .001),
        'physically_tested_closed_loop': False,
        'caution': 'Imitation accuracy does not establish on-robot success; do not replace the working expert without simulator evaluation.'
    }
    arrays = {'format': np.asarray(FORMAT), 'skill': np.asarray(skill),
              'layers': np.asarray(len(weights)), 'x_mean': xmean,
              'x_scale': xscale, 'y_mean': ymean, 'y_scale': yscale,
              'action_min': y.min(0).astype(np.float64),
              'action_max': y.max(0).astype(np.float64),
              'observed_x_min': xmin.astype(np.float64),
              'observed_x_max': xmax.astype(np.float64)}
    for i,(w,b) in enumerate(weights): arrays[f'w{i}'] = w; arrays[f'b{i}'] = b
    return arrays, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='*', default=[], help='Existing expert demonstration .npz files')
    p.add_argument('--data-dir', default=None, help='Folder containing expert_*.npz (e.g. expert_demos)')
    p.add_argument('--out', required=True, help='NEW output directory (will not overwrite)')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--arrival-max-error', type=float, default=.075,
                   help='Legacy V8.2 first release sample maximum distance from goal')
    args = p.parse_args()
    if args.epochs < 1: p.error('--epochs must be >=1')
    if not .045 <= args.arrival_max_error <= .10: p.error('--arrival-max-error must be .045 to .10m')
    paths = [Path(x).expanduser().resolve() for x in args.data]
    if args.data_dir:
        folder = Path(args.data_dir)
        if not folder.is_dir(): p.error(f'Demo folder not found: {folder}')
        paths.extend(folder.glob('expert_*.npz'))
    paths = list(dict.fromkeys(str(p.resolve()) for p in paths))
    if len(paths) < 2:
        p.error('Need >=2 expert NPZ demonstrations. Upload/run the existing Ubuntu traces, not .log files.')
    verified, info, skipped = [], [], []
    for f in paths:
        try:
            item = read_episode(Path(f), args.arrival_max_error)
        except (OSError, KeyError, ValueError) as exc:
            print('SKIP invalid episode:', f, str(exc), flush=True)
            skipped.append({'file': str(f), 'reason': str(exc)})
            continue
        if item is None:
            skipped.append({'file': str(f), 'reason': 'no verified ground contact and final XY placement at goal'})
            continue
        ep, meta = item
        verified.append(ep)
        info.append(meta)
    if len(verified) < 2:
        p.error(f'Only {len(verified)} verified ground-placement episodes, need >=2. Read the skipped messages.')
    # Repeated deterministic simulations do not produce independent validation.
    digests = [hashlib.sha256(e['carry'][0].tobytes()+e['carry'][1].tobytes()+
                              e['release'][0].tobytes()).hexdigest() for e in verified]
    duplicates = len(set(digests)) != len(digests)
    rng = np.random.default_rng(args.seed)
    ix = rng.permutation(len(verified)); nval = max(1, int(round(len(ix)*.2)))
    train_eps = [verified[int(i)] for i in ix[nval:]]
    val_eps = [verified[int(i)] for i in ix[:nval]]
    output = Path(args.out)
    if output.exists(): p.error(f'Output already exists: {output}')
    output.mkdir(parents=True)
    report = {
        'system': 'HYBRID: pretrained ML pickup/arm/V4 gripper + scripted expert waypoint and release',
        'success_definition': 'verified grasp/carry and physical release at waypoint; landing XY not required',
        'existing_base_arm_grasp_weights_preserved': True,
        'successful_demonstrations': info,
        'count': len(verified), 'skipped': skipped,
        'train_episode_count': len(train_eps), 'heldout_episode_count': len(val_eps),
        'exact_duplicate_episode_content': duplicates,
        'validation_warning': 'Repeated/identical starts can inflate validation accuracy; collect varied episodes before making generalization claims.',
        'seed': args.seed, 'epochs_max': args.epochs, 'skills': {},
        'trained_candidate_models_not_integrated_into_working_expert': True,
    }
    for skill in ('carry','release'):
        arrays, stats = train_skill(skill, train_eps, val_eps, args.epochs, args.seed, args.device)
        with (output/f'{skill}_candidate_bc.npz').open('xb') as f:
            np.savez_compressed(f, **arrays)
        report['skills'][skill] = stats
        print(skill, 'validation MAE', stats['heldout_mae_per_action'], flush=True)
    (output/'training_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('TRAINED CANDIDATE WEIGHTS:', output)
    print('NOT YET VALIDATED CLOSED-LOOP IN ROBOTNIK. Working expert unchanged.')
    if duplicates: print('WARNING: Duplicate expert episodes; reported holdout may not be independent.')


if __name__ == '__main__':
    main()
