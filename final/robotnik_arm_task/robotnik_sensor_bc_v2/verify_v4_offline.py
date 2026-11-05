#!/usr/bin/env python3
"""Checks V4 NN inference from rest in a simplified offline finger servo.

This does NOT use PyBullet, measure contact, or validate physical grasp.
Run from the directory containing sensor_bc_v4.py and the V4 checkpoint.
"""
import argparse
import json
import numpy as np
from sensor_bc_v4 import SensorPolicy


def run(policy, delta_xyz, seconds=5., dt=.001, motor_speed=.5):
    q = np.full(6, 0.1, dtype=np.float64)
    qd = np.zeros(6)
    starting = q.copy()
    for i in range(int(seconds/dt)):
        x = np.r_[delta_xyz, q, qd, [0., 0.]]
        target = policy.predict(x)
        if not np.isfinite(target).all():
            raise AssertionError('Invalid model targets')
        newq = q + np.clip(target-q, -motor_speed*dt, motor_speed*dt)
        qd = (newq-q)/dt
        q = newq
    # This is a motor-rate approximation, not a test of forces or contacts.
    return dict(delta_xyz=list(map(float,delta_xyz)), initial_q=starting.tolist(),
                end_q=np.round(q,5).tolist(),
                moves_past_old_stall=bool(np.min(q) > .45),
                within_training_target_range=bool(np.all(q <= policy.high+1e-7)))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='grasp_sensor_residual_bc_v4.npz')
    a=ap.parse_args()
    p=SensorPolicy(a.checkpoint)
    scenarios=[(0.,0.,0.),(-.008,.026,.024),(.01,-.03,.02),(.04,.04,-.03)]
    results=[run(p,np.array(r)) for r in scenarios]
    print(json.dumps(dict(test='simplified offline servo, not PyBullet contact',
                          scenarios=results), indent=2))
    if not all(r['moves_past_old_stall'] and r['within_training_target_range'] for r in results):
        raise SystemExit('FAIL: offline progress test did not pass')
    print('PASS: Learned V4 residual actions continue from rest in the simplified servo.')
