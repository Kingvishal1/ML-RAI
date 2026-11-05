"""Seven-joint neural return-home policy. Only the actuator and safety bounds are classical.

Observation: [goal_q - measured_q (7), measured_qdot (7)].
Network output: joint reference increment (7), in radians per simulation step.
"""
from pathlib import Path
import numpy as np

class LearnedReturnHomePolicy:
    def __init__(self, layers, obs_mean, obs_std, max_step):
        self.layers = layers
        self.obs_mean = np.asarray(obs_mean, dtype=np.float64)
        self.obs_std = np.maximum(np.asarray(obs_std, dtype=np.float64), 1e-5)
        self.max_step = float(max_step)

    @classmethod
    def load(cls, checkpoint):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Return-home checkpoint not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(data['format']) != 'robotnik_return_home_bc_v1':
                raise ValueError(f"Unexpected return-home checkpoint format: {path}")
            n_layers = int(data['num_layers'])
            layers = [(np.asarray(data[f'w{i}'], dtype=np.float64),
                       np.asarray(data[f'b{i}'], dtype=np.float64))
                      for i in range(n_layers)]
            policy = cls(layers, data['obs_mean'], data['obs_std'], float(data['max_step']))
        if policy.obs_mean.shape != (14,) or layers[-1][1].shape != (7,):
            raise ValueError('Invalid return-home network dimensions')
        return policy

    def predict_delta(self, measured_q, measured_qdot, home_q):
        observation = np.concatenate([
            np.asarray(home_q, dtype=np.float64) - np.asarray(measured_q, dtype=np.float64),
            np.asarray(measured_qdot, dtype=np.float64)])
        if observation.shape != (14,) or not np.all(np.isfinite(observation)):
            raise RuntimeError('Invalid neural return-home observations')
        x = (observation - self.obs_mean) / self.obs_std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers)-1:
                x = np.tanh(x)
        if x.shape != (7,) or not np.all(np.isfinite(x)):
            raise RuntimeError('Invalid neural return-home inference')
        return np.clip(x, -self.max_step, self.max_step)
