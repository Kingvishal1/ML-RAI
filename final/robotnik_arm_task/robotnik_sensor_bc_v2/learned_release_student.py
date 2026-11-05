"""Neural inference for the V20 one-ball placement/release student.

The network outputs seven bounded arm-joint increments, six finger target angles,
and one learned probability of release. No expert-release fallback is present.
"""
from pathlib import Path
import hashlib
import numpy as np


class LearnedReleaseStudent:
    def __init__(self, file):
        if not file or not Path(file).is_file():
            raise FileNotFoundError('Trained release checkpoint missing: train before testing')
        self.path = Path(file)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(file, allow_pickle=False) as z:
            if (str(z['format']) != 'v20_neural_release_bc_v1' or
                str(z['skill']) != 'placement' or not bool(z['trained'])):
                raise ValueError('Not a verified trained release-student checkpoint')
            self.teacher_source = str(z['teacher_source'])
            self.xmean = z['xmean'].astype(np.float64)
            self.xstd = z['xstd'].astype(np.float64)
            self.layers = [(z[f'w{i}'].astype(np.float64),
                            z[f'b{i}'].astype(np.float64)) for i in range(int(z['layers']))]
        if self.xmean.shape != (37,) or self.xstd.shape != (37,) or np.any(self.xstd <= 0):
            raise ValueError('Wrong release-student feature normalization shape')
        last = 37
        for w,b in self.layers:
            if w.shape[0] != last or b.shape != (w.shape[1],):
                raise ValueError('Malformed trained network layer')
            last = w.shape[1]
        if last != 14 or not all(np.isfinite(x).all() for x in
                [self.xmean, self.xstd] + [a for wb in self.layers for a in wb]):
            raise ValueError('Malformed release weights; refusing scripted fallback')

    def predict(self, obs):
        x = np.asarray(obs,dtype=np.float64)
        if x.shape != (37,) or not np.isfinite(x).all():
            raise ValueError('Expected 37 finite release observations')
        h = np.clip((x-self.xmean)/self.xstd,-10,10)
        for i,(w,b) in enumerate(self.layers):
            h=h@w+b
            if i+1 != len(self.layers):h=np.tanh(h)
        action = np.concatenate([np.clip(h[:7],-0.15,0.15),
                                 np.clip(h[7:13],0.0,2.0),
                                 [1/(1+np.exp(-np.clip(h[13],-50,50)))]])
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError('Invalid neural release output')
        return action
