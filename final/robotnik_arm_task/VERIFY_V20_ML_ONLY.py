#!/usr/bin/env python3
"""Static and numeric checks. These do not establish closed-loop success."""
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'robotnik_sensor_bc_v2'))
from robotnik_learned_task_policy import carry_features, LearnedTaskPolicy
from placement_policy import placement_features
src=(root/'robotnik_sensor_bc_v2/robotnik_v20_ml_only_two_ball.py').read_text()
assert 'choices=["bc"], default="bc"' in src
assert "choices=['bc'], default='bc'" in src
assert 'V20_ML_ONLY_CARRY_ARM' in src
assert "torque_arm = np.clip(learned_carry_action[2:9]" in src
assert 'V20_ML_ONLY_RELEASE_START' in src
assert 'no latched neural release action' in src
assert 'target_finger_angles[:] = expert_open_fingers' not in src
assert 'V8_EXPERT_RELEASE_START' not in src
assert "scripted high release is disabled" in src
x=carry_features(np.zeros(17),np.zeros(17),np.array([1.,3.,1.]),
     np.zeros(3),np.array([1.,3.,.95]),np.array([3.,3.5]),
     0.,np.zeros(3),np.zeros(3),np.array([1.,3.]))
assert x.shape==(25,) and np.isfinite(x).all()
y=placement_features(np.zeros(17),np.zeros(17),np.array([1.,3.,1.]),
    np.zeros(3),np.array([1.,3.,.95]),np.array([3.,3.5]),.55,0.,0.)
assert y.shape==(37,) and np.isfinite(y).all()
try:
    LearnedTaskPolicy(root/'definitely_no_trained_checkpoint.npz','carry_full')
except FileNotFoundError:
    pass
else:
    raise AssertionError('Missing trained weights must fail closed')
print('PASS: ML-only code routes carry base+arm actions and release through neural policies;')
print('PASS: 25- and 37-feature sensor contracts; missing checkpoints fail closed.')
print('NOT a physics success check. Training and PyBullet testing are still required.')
