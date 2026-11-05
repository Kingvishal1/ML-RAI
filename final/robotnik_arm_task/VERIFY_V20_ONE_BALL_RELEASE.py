#!/usr/bin/env python3
"""Check real checkpoint loading and neural action dimensions. Not physics verification."""
import argparse,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkpoint');a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parent/'robotnik_sensor_bc_v2'))
from learned_release_student import LearnedReleaseStudent
if a.checkpoint:
    model=LearnedReleaseStudent(a.checkpoint)
    action=model.predict(np.zeros(37,dtype=float))
    assert action.shape==(14,) and np.isfinite(action).all()
    assert 0<=action[-1]<=1 and np.all((action[7:13]>=0)&(action[7:13]<=2))
    print('PASS: trained neural checkpoint loads; 37 inputs -> 14 outputs; source:',model.teacher_source)
else:
    import tempfile
    try:LearnedReleaseStudent('/definitely/missing/weights.npz')
    except FileNotFoundError:pass
    else:raise AssertionError('Missing checkpoint must not enable scripted fallback')
    print('PASS: missing checkpoint is rejected, no scripted fallback')
