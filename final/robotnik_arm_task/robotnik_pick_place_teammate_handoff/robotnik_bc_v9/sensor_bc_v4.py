#!/usr/bin/env python3
"""Robotnik V4: learned, observation-conditioned residual finger-position BC.

Training source is a *single virtual-attachment teacher demonstration*, not a
verified physical-grasp demonstration. This policy is an experimental contact-
seeking baseline. Its 6 outputs are finger-position targets obtained by adding
learned, NN-predicted increments to the sensed joint angles. There is NO
scripted finger sweep, manually imposed minimum increment, contact override,
virtual attachment, or teacher call at inference.

The existing 17 sensor features exclude phase, timer, and virtual-grasp flags.
During training, augmented stopped/perturbed states improve robustness to the
failure observed with the original absolute-action BC model. Augmented samples
are synthetic (not new physical demonstrations) and must not be considered
independent evaluation data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = 'robotnik_sensor_residual_bc_v4'
SOURCE_FORMAT = 'robotnik_sensor_bc_v2'
FINGER_SLICE = slice(3, 9)
# These floors are in SI units and prevent tiny sensor variance amplifying noise.
X_SCALE_FLOORS = np.array([.03]*3 + [.10]*6 + [.50]*6 + [.05]*2)


def sensor_features(ee, ball, finger_q, finger_qd, velocity_xy, bearing):
    """Same 17 inputs, in same order, as the existing sensor_bc.py."""
    c, s = np.cos(bearing), np.sin(bearing)
    vx, vy = velocity_xy
    delta = np.asarray(ball) - np.asarray(ee)
    return np.asarray([c*delta[0]+s*delta[1],
                       -s*delta[0]+c*delta[1], delta[2],
                       *finger_q, *finger_qd,
                       c*vx+s*vy, -s*vx+c*vy], dtype=np.float64)


def convert_legacy(x):
    """Legacy 24-dim observation -> original 17 sensor features."""
    if x.ndim != 2 or x.shape[1] != 24:
        raise ValueError('Expected legacy 24-feature observation array')
    return np.stack([sensor_features(np.zeros(3), r[:3], r[3:9], r[9:15],
                                    r[15:17], np.arctan2(r[18], r[17]))
                     for r in x])


class SensorPolicy:
    """Fully learned residual action: target = measured_q + MLP(sensor_state)."""
    def __init__(self, filename):
        with np.load(filename, allow_pickle=False) as z:
            if str(z['format']) != FORMAT:
                raise ValueError('Not a V4 residual finger BC model')
            self.x_mean = np.asarray(z['x_mean'], dtype=np.float64)
            self.x_scale = np.asarray(z['x_scale'], dtype=np.float64)
            self.delta_mean = np.asarray(z['delta_mean'], dtype=np.float64)
            self.delta_scale = np.asarray(z['delta_scale'], dtype=np.float64)
            self.low = np.asarray(z['action_min'], dtype=np.float64)
            self.high = np.asarray(z['action_max'], dtype=np.float64)
            self.layers = [(z[f'w{i}'], z[f'b{i}']) for i in range(int(z['layers']))]
        if (self.x_mean.shape != (17,) or self.x_scale.shape != (17,) or
                self.low.shape != (6,) or self.high.shape != (6,) or
                self.delta_mean.shape != (6,) or self.delta_scale.shape != (6,)):
            raise ValueError('Inconsistent checkpoint tensor dimensions')
        if np.any(self.x_scale <= 0) or np.any(self.delta_scale <= 0):
            raise ValueError('Invalid normalisation scales')

    def predict_delta(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (17,) or not np.isfinite(x).all():
            raise ValueError('Expected finite 17-feature observation')
        h = (x - self.x_mean) / self.x_scale
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        delta = h * self.delta_scale + self.delta_mean
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite neural-network action')
        return delta

    def predict(self, features, bounded=True):
        x = np.asarray(features, dtype=np.float64)
        delta = self.predict_delta(x)
        target = x[FINGER_SLICE] + delta
        # Only physical / demonstrated action bounds; no hardcoded squeeze.
        return np.clip(target, self.low, self.high) if bounded else target


def read_legacy_demo(path):
    with np.load(path, allow_pickle=False) as z:
        if 'obs' not in z or 'action' not in z:
            raise ValueError(f'{path} lacks obs/action arrays')
        legacy = np.asarray(z['obs'], dtype=np.float64)
        actions = np.asarray(z['action'], dtype=np.float64)
    if legacy.ndim != 2 or legacy.shape[1] != 24 or actions.shape != (len(legacy), 10):
        raise ValueError(f'Incompatible demonstration shape in {path}')
    if not np.isfinite(legacy).all() or not np.isfinite(actions).all():
        raise ValueError('Nonfinite demonstration')
    closing = legacy[:, 21] > .5
    holding = legacy[:, 22] > .5
    if closing.sum() < 30:
        raise ValueError('Need at least 30 original closing frames')
    return convert_legacy(legacy), actions[:, :6], closing, holding, legacy


def make_augmented(x, targets, closing, holding, seed=42, augment=8):
    """Create explicit *synthetic* state-perturbation samples for the BC teacher.

    Closing labels use the observed expert incremental target, applied to a
    perturbed current joint configuration. Holding labels retain observed teacher
    targets. The robot/ball/contact physics is NOT simulated by this function.
    """
    rng = np.random.default_rng(seed)
    close_ids = np.flatnonzero(closing)
    hold_ids = np.flatnonzero(holding)
    # Avoid a large number of duplicate holding frames swamping actual closing.
    if len(hold_ids) > 40:
        hold_ids = hold_ids[np.linspace(0, len(hold_ids)-1, 40, dtype=int)]
    chosen = np.concatenate((close_ids, hold_ids))
    kinds = np.concatenate((np.ones(len(close_ids), dtype=bool),
                            np.zeros(len(hold_ids), dtype=bool)))
    if len(chosen) == 0:
        raise ValueError('Empty dataset')
    xx, dd = [], []
    hi = targets.max(axis=0)
    lo = np.zeros(6)
    for index, is_closing in zip(chosen, kinds):
        source = x[index].copy()
        source_q = source[FINGER_SLICE]
        source_target = targets[index]
        for aug_id in range(augment + 1):
            feat = source.copy()
            if aug_id:
                # Disturb grasp geometry and approach velocity; this is domain
                # randomization, not a physical success demonstration.
                feat[:3] += rng.uniform(-.045, .045, size=3)
                feat[3:9] = np.clip(source_q + rng.normal(0, .035, size=6), lo, hi)
                # 75% stopped / slowed samples: addresses closed-loop freeze.
                if rng.random() < .75:
                    feat[9:15] = 0.0
                else:
                    feat[9:15] *= rng.uniform(.0, 1.2)
                feat[15:17] += rng.normal(0, .02, size=2)
            if is_closing:
                # Supervised expert DELTA; not an inference-time sweep.
                label_target = feat[3:9] + (source_target - source_q)
            else:
                label_target = source_target.copy()
            label_target = np.clip(label_target, lo, hi)
            xx.append(feat)
            dd.append(label_target - feat[3:9])
    return np.asarray(xx), np.asarray(dd), dict(
        original_close_frames=len(close_ids), original_hold_frames=len(hold_ids),
        synthetic_per_original=int(augment), total_train_examples=len(xx))


def train(data_paths, out, seed=42, augment=8, max_iter=350):
    from sklearn.neural_network import MLPRegressor
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Existing checkpoint protected: {out}')
    all_x, all_y, all_c, all_h = [], [], [], []
    hashes = {}
    for path in data_paths:
        x, y, closing, holding, _ = read_legacy_demo(path)
        all_x.append(x); all_y.append(y); all_c.append(closing); all_h.append(holding)
        hashes[Path(path).name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    x = np.concatenate(all_x); y = np.concatenate(all_y)
    closing = np.concatenate(all_c); holding = np.concatenate(all_h)
    augmented_x, augmented_delta, description = make_augmented(
        x, y, closing, holding, seed=seed, augment=augment)
    mean = augmented_x.mean(axis=0)
    scale = np.maximum(augmented_x.std(axis=0), X_SCALE_FLOORS)
    delta_mean = augmented_delta.mean(axis=0)
    # A floor scales the tiny expert increments to a learnable range.
    delta_scale = np.maximum(augmented_delta.std(axis=0), .02)
    net = MLPRegressor(hidden_layer_sizes=(64,64), activation='tanh',
                       solver='adam', random_state=seed, alpha=5e-4,
                       learning_rate_init=.001, max_iter=max_iter,
                       batch_size=128, n_iter_no_change=40, tol=1e-7,
                       early_stopping=False)
    net.fit((augmented_x-mean)/scale, (augmented_delta-delta_mean)/delta_scale)
    params = dict(format=np.asarray(FORMAT), layers=np.asarray(len(net.coefs_)),
                  x_mean=mean, x_scale=scale,
                  delta_mean=delta_mean, delta_scale=delta_scale,
                  action_min=np.min(y,axis=0), action_max=np.max(y,axis=0))
    for i, (w,b) in enumerate(zip(net.coefs_,net.intercepts_)):
        params[f'w{i}'] = w
        params[f'b{i}'] = b
    with out.open('xb') as f:
        np.savez_compressed(f, **params)
    policy = SensorPolicy(out)
    expert_delta = y[closing] - x[closing,3:9]
    predicted_delta = np.vstack([policy.predict_delta(row) for row in x[closing]])
    # Unaugmented source errors are *training-set checks*, not held-out tests.
    close_mae = np.abs(predicted_delta-expert_delta).mean(axis=0)
    # Probe the formerly stalling case at 0 measured qd using a real source q.
    stationary = x[np.flatnonzero(closing)[12]].copy()
    stationary[9:15] = 0
    stationary_delta = policy.predict_delta(stationary)
    report = dict(
        checkpoint=out.name, format=FORMAT, source_files=hashes,
        scope='contact-seeking residual finger BC ONLY, reverse disabled',
        supervised_source='virtual-attachment teacher; NO verified physical grasp',
        original_frames=len(x), original_close_frames=int(closing.sum()),
        original_hold_frames=int(holding.sum()),
        synthetic_training_augmentation=True, augmentation=description,
        independent_episode_validation=False, physical_grasp_validated=False,
        closed_loop_validated=False, network='MLP tanh 17-64-64-6; predicts delta positions',
        stopped_state_probe_delta_rad=np.round(stationary_delta,6).tolist(),
        unaugmented_training_close_delta_mae_rad=np.round(close_mae,6).tolist(),
        training_iterations=int(net.n_iter_))
    out.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',nargs='+',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--augment',type=int,default=8)
    ap.add_argument('--max_iter',type=int,default=350)
    a = ap.parse_args()
    train(a.data, a.out, a.seed, a.augment, a.max_iter)
