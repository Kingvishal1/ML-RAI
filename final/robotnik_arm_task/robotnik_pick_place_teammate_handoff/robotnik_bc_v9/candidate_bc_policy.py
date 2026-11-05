#!/usr/bin/env python3
"""NumPy inference for a BC student trained by train_from_verified_expert.py.

Not an expert controller. Predicts a motor action from the saved MLP weights.
Does not decide task phases or replace PyBullet physics validations.
"""
from pathlib import Path
import hashlib
import numpy as np


class CandidatePolicy:
    def __init__(self, path, expected_skill=None):
        if not path:
            raise ValueError('BC checkpoint path is required')
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as z:
            if str(z['format']) != 'robotnik_expert_bc_mlp_v1':
                raise ValueError('Not a Robotnik BC MLP v1 checkpoint')
            self.skill = str(z['skill'])
            if self.skill not in ('carry', 'release') or (expected_skill and self.skill != expected_skill):
                raise ValueError(f'Wrong model skill: expected {expected_skill}, got {self.skill}')
            self.x_mean = np.asarray(z['x_mean'], dtype=np.float64).copy()
            self.x_scale = np.asarray(z['x_scale'], dtype=np.float64).copy()
            self.y_mean = np.asarray(z['y_mean'], dtype=np.float64).copy()
            self.y_scale = np.asarray(z['y_scale'], dtype=np.float64).copy()
            self.y_min = np.asarray(z['action_min'], dtype=np.float64).copy()
            self.y_max = np.asarray(z['action_max'], dtype=np.float64).copy()
            layers = int(z['layers'])
            if layers < 1 or layers > 8:
                raise ValueError(f'Unexpected layers: {layers}')
            self.weights = [(np.asarray(z[f'w{i}'], dtype=np.float64).copy(),
                             np.asarray(z[f'b{i}'], dtype=np.float64).copy()) for i in range(layers)]
        nx,ny = (5,2) if self.skill == 'carry' else (17,6)
        if not (self.x_mean.shape == self.x_scale.shape == (nx,) and
                self.y_mean.shape == self.y_scale.shape == self.y_min.shape == self.y_max.shape == (ny,)):
            raise ValueError('BC model feature/action dimensions do not match recorded expert data')
        if (np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0) or
                np.any(self.y_min > self.y_max)):
            raise ValueError('Invalid normalization or output bounds in BC checkpoint')
        size = nx
        for i,(w,b) in enumerate(self.weights):
            if w.ndim != 2 or w.shape[0] != size or b.shape != (w.shape[1],):
                raise ValueError(f'Invalid weight shape at layer {i}')
            size = w.shape[1]
        if size != ny:
            raise ValueError(f'Invalid BC output dimensionality: {size}')
        for arr in (self.x_mean,self.x_scale,self.y_mean,self.y_scale,
                    self.y_min,self.y_max,*[item for wb in self.weights for item in wb]):
            if not np.isfinite(arr).all():
                raise ValueError('Nonfinite values in BC checkpoint')

    def predict(self, observations):
        x = np.asarray(observations, dtype=np.float64)
        if x.shape[-1] != self.x_mean.size or not np.isfinite(x).all():
            raise ValueError(f'Wrong/nonfinite inputs for {self.skill} BC; expected {self.x_mean.size} features')
        x = (x - self.x_mean) / self.x_scale
        for i,(weight,bias) in enumerate(self.weights):
            x = x @ weight + bias
            if i != len(self.weights)-1:
                x = np.tanh(x)
        action = np.clip(x*self.y_scale + self.y_mean, self.y_min, self.y_max)
        if not np.isfinite(action).all():
            raise ValueError('BC policy produced a nonfinite action')
        return action
