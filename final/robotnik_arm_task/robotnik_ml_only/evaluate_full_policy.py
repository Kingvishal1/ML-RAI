"""Deterministic full-actor test. No scripted motor phases or virtual grasp."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from full_policy import load_actor,encode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed',type=int,default=1042)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--out',default='full_policy_eval.json')
    a=p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    if a.episodes<1:p.error('episodes must be positive')
    torch.set_num_threads(1);m,ck=load_actor(a.checkpoint)
    print('Checkpoint status:',ck['status'],'; scene: weightless dynamic ball')
    from robotnik_env import RobotnikEnv
    env=RobotnikEnv(headless=a.headless,seed=a.seed,control_steps=int(ck['control_steps']))
    results=[]
    try:
        np.testing.assert_allclose(env.low,ck['action_low'].numpy())
        np.testing.assert_allclose(env.high,ck['action_high'].numpy())
        for ep in range(a.episodes):
            obs=env.reset();total=0.
            while True:
                with torch.no_grad():act=m.action(torch.as_tensor(encode(obs))).numpy()
                obs,r,term,trunc,info=env.step(act);total+=r
                if not a.headless:time.sleep(env.dt*env.skip)
                if term or trunc:break
            result=dict(episode=ep,return_=total,candidate_success=info['success'],
                safety_stop=info['safety_stop'],safety_reasons=info['safety_reasons'],
                arm_joint_speeds=np.round(info['joint_velocities'],3).tolist(),
                ball_goal_distance=info['ball_goal_distance'],
                hold_seconds=info['hold_seconds'],finger_groups=info['finger_groups'],
                reverse_distance=info['reverse_distance'])
            results.append(result);print(json.dumps(result),flush=True)
    finally:
        env.close()
        Path(a.out).write_text(json.dumps(dict(checkpoint=a.checkpoint,seed=a.seed,
            scene='weightless_dynamic_ball',results=results,
            note='Candidate success is a task metric; inspect grasp visually and test robustness.'),indent=2))
if __name__=='__main__':main()
