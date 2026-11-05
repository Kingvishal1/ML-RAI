"""Sensor-only close/reverse behavioural cloning. NOT a full-task policy.
Inference needs NumPy only; training also needs scikit-learn.
No timer, scripted phase, distance-since-transition or virtual grasp flag as inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np

FORMAT = 'robotnik_sensor_bc_v2'

def sensor_features(ee, ball, finger_q, finger_qd, velocity_xy, bearing):
    c, s = np.cos(bearing), np.sin(bearing)
    vx, vy = velocity_xy
    delta = np.asarray(ball) - np.asarray(ee)
    return np.asarray([c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1], delta[2],
                       *finger_q, *finger_qd, c*vx+s*vy, -s*vx+c*vy], dtype=np.float64)

def convert_legacy(x):
    # Legacy order: ball-ee xyz, finger q(6), qd(6), world vx/vy,
    # cos(yaw), sin(yaw), distance, timer, scripted phase one-hot(3).
    return np.stack([sensor_features(np.zeros(3), r[:3], r[3:9], r[9:15],
                                    r[15:17], np.arctan2(r[18], r[17])) for r in x])

class SensorPolicy:
    def __init__(self, filename):
        with np.load(filename, allow_pickle=False) as z:
            if str(z['format']) != FORMAT:
                raise ValueError('Not a sensor BC v2 model')
            self.x_mean, self.x_scale = z['x_mean'], z['x_scale']
            self.y_mean, self.y_scale = z['y_mean'], z['y_scale']
            self.low, self.high = z['action_min'], z['action_max']
            self.layers = [(z[f'w{i}'], z[f'b{i}']) for i in range(int(z['layers']))]
        if self.x_mean.shape != (17,) or self.low.shape != (10,):
            raise ValueError('Invalid model dimensions')

    def predict(self, x, bounded=True):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != 17 or not np.isfinite(x).all():
            raise ValueError('Expected finite 17-dimensional sensor input')
        h = (x-self.x_mean)/self.x_scale
        for i, (w,b) in enumerate(self.layers):
            h = h@w+b
            if i < len(self.layers)-1:
                h = np.tanh(h)
        y = h*self.y_scale+self.y_mean
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite policy output')
        # Dataset-support clamp, not a learned safety guarantee.
        return np.clip(y, self.low, self.high) if bounded else y

def load_data(paths):
    result=[]
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            x, y = z['obs'].copy(), z['action'].copy()
        if x.ndim != 2 or x.shape[1] != 24 or y.shape != (len(x),10) or len(x)<60:
            raise ValueError(f'Invalid legacy demonstration: {p}')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite training data')
        result.append((convert_legacy(x),y,x))
    return result

def train(paths, out, epochs=600):
    from sklearn.neural_network import MLPRegressor
    out=Path(out)
    if out.exists():
        raise FileExistsError(out)
    episodes=load_data(paths)
    if len(episodes)>1:
        x=np.concatenate([e[0] for e in episodes[:-1]])
        y=np.concatenate([e[1] for e in episodes[:-1]])
        xv,yv,_=episodes[-1]
        split='last supplied episode held out; independence depends on its collection'
    else:
        xx,yy,legacy=episodes[0]
        tr,va=[],[]
        for phase_col in (21,22):
            ids=np.flatnonzero(legacy[:,phase_col]>.5)
            if len(ids)<30:
                raise ValueError('Need closing and reversing samples')
            cut=int(.75*len(ids))
            tr.extend(ids[:cut-10]); va.extend(ids[cut:])
        x,y=xx[tr],yy[tr]; xv,yv=xx[va],yy[va]
        split='single trajectory, chronological holdout within each phase; 10-frame gap; NOT episode generalization'
    xm=x.mean(0)
    # Physical scale floors prevent millimetre sensor changes becoming huge inputs.
    xs=np.maximum(x.std(0),[.03]*3+[.1]*6+[.5]*6+[.05]*2)
    ym=y.mean(0); ys=np.maximum(y.std(0),.1)
    net=MLPRegressor(hidden_layer_sizes=(64,64),activation='tanh',solver='lbfgs',
                     max_iter=epochs,max_fun=30000,alpha=.01,random_state=42,tol=1e-7)
    net.fit((x-xm)/xs,(y-ym)/ys)
    pred=net.predict((xv-xm)/xs)*ys+ym
    params=dict(format=np.asarray(FORMAT),layers=np.asarray(len(net.coefs_)),
        x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys,
        action_min=y.min(0),action_max=y.max(0))
    for i,(w,b) in enumerate(zip(net.coefs_,net.intercepts_)):
        params[f'w{i}']=w; params[f'b{i}']=b
    # Open exclusively; never overwrite an existing model.
    with out.open('xb') as f:
        np.savez_compressed(f,**params)
    policy=SensorPolicy(out)
    np.testing.assert_allclose(policy.predict(xv,bounded=False),pred,rtol=1e-10,atol=1e-10)
    baseline=np.broadcast_to(ym,yv.shape)
    report=dict(format=FORMAT,scope='closing/reversing only; virtual-attachment teacher data',
        input_features=17,output_actions=10,train_frames=len(x),holdout_frames=len(xv),
        split=split,seed=42,iterations=int(net.n_iter_),
        raw_holdout_mae=np.abs(pred-yv).mean(0).tolist(),
        bounded_holdout_mae=np.abs(policy.predict(xv)-yv).mean(0).tolist(),
        constant_baseline_mae=np.abs(baseline-yv).mean(0).tolist(),
        clamp_fraction=float(np.mean(np.abs(policy.predict(xv)-pred)>1e-8)),
        physical_grasp_validated=False,closed_loop_validated=False,
        full_task_policy=False,learned_stop=False,
        feature_scale_floors='position .03m; finger angle .1rad; joint velocity .5rad/s; base velocity .05m/s',
        source_files=[Path(p).name for p in paths])
    out.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=600)
    a=p.parse_args(); train(a.data,a.out,a.epochs)
