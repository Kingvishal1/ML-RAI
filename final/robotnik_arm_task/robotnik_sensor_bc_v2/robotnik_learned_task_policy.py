"""V20 genuine inference-only task policies. Never generates expert motor actions.

Carry: 25 sensor features -> (v, yaw_rate, seven arm torques)
Placement: 37 sensor features -> (seven arm target deltas,
             six finger targets, learned release probability)

The joint-target servo is conventional low-level actuation, not a programmed
trajectory. The task-level arm movement / wheel velocities / release decision
come exclusively from loaded neural checkpoints.
"""
from pathlib import Path
import hashlib
import numpy as np

FORMAT='robotnik_v20_ml_only_v1'
DIMS={'carry_full':(25,9),'placement_full':(37,14)}

def carry_features(q,qd,ball,ball_velocity,hand,goal_xy,yaw,
                   base_linear_velocity,base_angular_velocity,carry_origin_xy):
    q=np.asarray(q,dtype=float); qd=np.asarray(qd,dtype=float)
    ball=np.asarray(ball,dtype=float); vel=np.asarray(ball_velocity,dtype=float)
    hand=np.asarray(hand,dtype=float); goal=np.asarray(goal_xy,dtype=float)
    base_vel=np.asarray(base_linear_velocity,dtype=float)
    base_w=np.asarray(base_angular_velocity,dtype=float)
    origin=np.asarray(carry_origin_xy,dtype=float)
    if any(x.shape != shape for x,shape in ((ball,(3,)),(vel,(3,)),
            (hand,(3,)),(goal,(2,)),(base_vel,(3,)),(base_w,(3,)),(origin,(2,)))):
        raise ValueError('Carry feature vector shape mismatch')
    if q.size < 11 or qd.size < 11:
        raise ValueError('Need 7 sensed arm joints and velocities')
    to_goal=goal-ball[:2]; remaining=float(np.linalg.norm(to_goal))
    desired_heading=float(np.arctan2(to_goal[1],to_goal[0])) if remaining>1e-6 else float(yaw)
    bearing_error=float(np.arctan2(np.sin(desired_heading-yaw),np.cos(desired_heading-yaw)))
    path=goal-origin; unit=path/max(np.linalg.norm(path),1e-8)
    travelled=ball[:2]-origin
    lateral=float(unit[0]*travelled[1]-unit[1]*travelled[0])
    forward=float(np.dot(base_vel[:2],[np.cos(yaw),np.sin(yaw)]))
    c,s=np.cos(yaw),np.sin(yaw)
    rot=np.array([[c,s,0.0],[-s,c,0.0],[0.0,0.0,1.0]])
    x=np.r_[[remaining,lateral,bearing_error,forward,base_w[2]],
        q[4:11],qd[4:11],rot@(ball-hand),rot@vel]
    if x.shape!=(25,) or not np.isfinite(x).all():
        raise ValueError('Carry observations must have 25 finite features')
    return x.astype(np.float32)

class LearnedTaskPolicy:
    def __init__(self,path,expected_skill):
        if expected_skill not in DIMS:
            raise ValueError(f'Unsupported ML skill {expected_skill}')
        self.path=Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f'ML checkpoint missing: {self.path}')
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path,allow_pickle=False) as z:
            if str(z['format']) != FORMAT:
                raise ValueError('Checkpoint is not a V20 ML-only trained policy')
            self.skill=str(z['skill'])
            if self.skill != expected_skill:
                raise ValueError(f'Wrong skill: expected {expected_skill}, received {self.skill}')
            if 'training_episodes' not in z or int(z['training_episodes'])<4:
                raise ValueError('Checkpoint lacks evidence of >=4 training episodes')
            if 'trained' not in z or not bool(z['trained'].item()):
                raise ValueError('Untrained weights are not accepted')
            self.x_mean=np.asarray(z['x_mean'],dtype=float).copy()
            self.x_scale=np.asarray(z['x_scale'],dtype=float).copy()
            self.y_mean=np.asarray(z['y_mean'],dtype=float).copy()
            self.y_scale=np.asarray(z['y_scale'],dtype=float).copy()
            self.y_min=np.asarray(z['action_min'],dtype=float).copy()
            self.y_max=np.asarray(z['action_max'],dtype=float).copy()
            self.weights=[]
            for i in range(int(z['layers'])):
                self.weights.append((np.asarray(z[f'w{i}'],dtype=float).copy(),
                                     np.asarray(z[f'b{i}'],dtype=float).copy()))
        ni,no=DIMS[self.skill]
        if any(v.shape!=(n,) for v,n in ((self.x_mean,ni),(self.x_scale,ni),
               (self.y_mean,no),(self.y_scale,no),(self.y_min,no),(self.y_max,no))):
            raise ValueError('Invalid feature/action dimensions')
        if np.any(self.x_scale<=0) or np.any(self.y_scale<=0) or np.any(self.y_min>self.y_max):
            raise ValueError('Invalid scaling or bounds')
        if len(self.weights) != 3:
            raise ValueError('Expected a trained 3-layer MLP')
        dims=[ni]+[w.shape[1] for w,b in self.weights]
        if dims[-1]!=no or any(w.shape!=(dims[i],dims[i+1]) or b.shape!=(dims[i+1],)
            for i,(w,b) in enumerate(self.weights)):
            raise ValueError('MLP architecture not compatible with this V20 controller')
        for arr in [self.x_mean,self.x_scale,self.y_mean,self.y_scale,self.y_min,self.y_max]:
            if not np.isfinite(arr).all():raise ValueError('Nonfinite model statistics')
        for w,b in self.weights:
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite learned weights')

    def predict(self,obs):
        x=np.asarray(obs,dtype=float)
        if x.shape != (DIMS[self.skill][0],) or not np.isfinite(x).all():
            raise ValueError('Invalid sensor observation for learned policy')
        h=(x-self.x_mean)/self.x_scale
        for i,(w,b) in enumerate(self.weights):
            h=h@w+b
            if i != len(self.weights)-1:h=np.tanh(h)
        y=h*self.y_scale+self.y_mean
        if self.skill=='placement_full':
            # The last output is a trained BCE release logit, not an expert flag.
            y[13]=1.0/(1.0+np.exp(-np.clip(h[13],-50.0,50.0)))
        y=np.clip(y,self.y_min,self.y_max)
        if not np.isfinite(y).all():raise ValueError('Nonfinite neural action')
        return y
