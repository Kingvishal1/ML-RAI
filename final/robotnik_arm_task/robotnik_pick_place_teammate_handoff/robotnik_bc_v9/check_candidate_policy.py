#!/usr/bin/env python3
"""Inspect and run offline inference on a candidate BC model. NOT a Robotnik runner."""
import argparse
import numpy as np

class CandidatePolicy:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['format']) != 'robotnik_expert_bc_mlp_v1':
                raise ValueError('Wrong model format')
            self.skill=str(z['skill'])
            self.xmean=z['x_mean'];self.xscale=z['x_scale'];self.ymean=z['y_mean'];self.yscale=z['y_scale']
            self.low=z['action_min'];self.high=z['action_max']
            self.layers=[(z[f'w{i}'],z[f'b{i}']) for i in range(int(z['layers']))]
    def predict(self,observations):
        x=np.asarray(observations,dtype=np.float64)
        if x.shape[-1] != len(self.xmean) or not np.isfinite(x).all():
            raise ValueError('Wrong feature dimensions or nonfinite values')
        h=(x-self.xmean)/self.xscale
        for i,(w,b) in enumerate(self.layers):
            h=h@w+b
            if i!=len(self.layers)-1:h=np.tanh(h)
        return np.clip(h*self.yscale+self.ymean,self.low,self.high)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model');p.add_argument('--inputs',nargs='+',type=float,required=True)
    a=p.parse_args();model=CandidatePolicy(a.model)
    print('Skill:',model.skill,'Candidate output:',model.predict(a.inputs).tolist())
