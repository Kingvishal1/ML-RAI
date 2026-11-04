#!/usr/bin/env python3
"""Train carry + placement BC from real, verified, recorded rollouts.

An entire episode is held out. Old logs, empty/duplicate traces, missing arm
labels, failed placements, and detached balls outside 5 cm are rejected.
Training metrics are imitation metrics, never a closed-loop success claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from train_from_verified_expert import SPECS, train_skill

SPECS['placement'] = ('placement_obs_37', 'placement_actions_14', 37, 14,
                      np.array([.10]*7 + [.25]*7 + [.10]*6 + [.50]*6 +
                               [.05]*3 + [.10]*3 + [.03]*3 + [.05, 1.0]))


def read_placement(path):
    with np.load(path, allow_pickle=False) as z:
        for flag in ('final_placement_verified', 'ground_contact_verified',
                     'ball_released_verified', 'no_virtual_attachment'):
            if flag not in z or not bool(np.asarray(z[flag]).item()):
                raise ValueError(f'{flag} absent or false')
        if not np.isfinite(float(z['final_ball_xy_error_m'])) or float(z['final_ball_xy_error_m']) > .05:
            raise ValueError('Final landing error exceeds 5 cm')
        if 'placement_format' not in z or str(z['placement_format']) != 'robotnik_placement_v1':
            raise ValueError('No lowering observations/actions; collect with the updated recorder')
        if str(z['placement_label_source']) != 'EXPERT_JOINT_REFERENCE_AND_RELEASE_DECISION':
            raise ValueError('Incorrect label provenance')
        eps = {}
        for skill in ('carry', 'placement'):
            xk, yk, nx, ny, _ = SPECS[skill]
            x, y = np.asarray(z[xk], dtype=np.float32), np.asarray(z[yk], dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != nx or y.shape != (len(x), ny) or len(x) < 20:
                raise ValueError(f'Invalid or insufficient {skill} frames')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite training values')
            eps[skill] = (x.copy(), y.copy())
        actions = eps['placement'][1]
        if not np.all(np.isin(actions[:, 13], [0., 1.])):
            raise ValueError('Release labels must be binary')
        if not ((actions[:, 13] == 0).sum() >= 10 and (actions[:, 13] == 1).sum() >= 5):
            raise ValueError('Both lowering/hold and release examples are required')
        if np.any(actions[:, 7:13] < -.01) or np.any(actions[:, 7:13] > 2.01):
            raise ValueError('Finger targets exceed physical limits')
        # Catch copied recordings even if filenames/metadata differ.
        digest = hashlib.sha256(b''.join(a.tobytes() for e in eps.values() for a in e)).hexdigest()
        return eps, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demos', default='expert_demos')
    ap.add_argument('--output', default='trained_placement_bc')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    accepted, hashes, audit = [], set(), []
    for path in sorted(Path(args.demos).glob('*.npz')):
        try:
            ep, digest = read_placement(path)
            if digest in hashes:
                raise ValueError('Duplicate episode')
            hashes.add(digest)
            accepted.append((path.name, ep))
            audit.append({'file': path.name, 'accepted': True, 'sha256_arrays': digest})
        except (ValueError, KeyError, OSError) as exc:
            audit.append({'file': path.name, 'accepted': False, 'reason': str(exc)})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'episode_audit.json').write_text(json.dumps(audit, indent=2))
    if len(accepted) < 4:
        raise SystemExit(f'Need at least 4 unique successful placement episodes; found {len(accepted)}. '
                         'No weights trained. See episode_audit.json.')
    if list(out.glob('*_bc.npz')):
        raise SystemExit('Output already contains weights; choose a new --output directory')
    order = np.random.default_rng(args.seed).permutation(len(accepted))
    nval = max(1, int(np.ceil(len(accepted)*.2)))
    val = [accepted[i] for i in order[:nval]]
    train = [accepted[i] for i in order[nval:]]
    reports = {'train_episodes': [n for n,_ in train], 'heldout_episodes': [n for n,_ in val],
               'closed_loop_verified': False, 'models': {}}
    for skill in ('carry', 'placement'):
        arrays, report = train_skill(skill, [e for _,e in train], [e for _,e in val],
                                     args.epochs, args.seed, args.device)
        arrays['feature_contract'] = np.asarray('robotnik_placement_v1')
        # Loading support bounds must include held-out values only for the
        # semantic binary score, never normalization statistics.
        if skill == 'placement':
            arrays['action_min'][13], arrays['action_max'][13] = 0., 1.
        name = 'carry_candidate_bc.npz' if skill == 'carry' else 'placement_bc.npz'
        with (out/name).open('xb') as f:
            np.savez_compressed(f, **arrays)
        reports['models'][skill] = report
        print(skill, 'held-out MAE:', report['heldout_mae_per_action'])
    (out/'training_report.json').write_text(json.dumps(reports, indent=2))
    print('CANDIDATE WEIGHTS SAVED:', out, 'Closed-loop simulator evaluation required.')


if __name__ == '__main__':
    main()
