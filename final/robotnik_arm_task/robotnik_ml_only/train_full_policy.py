"""Offline approach warm start followed by on-policy PPO for all 17 motor commands.
Requires the user's existing PyTorch/RoboEnv environment. No checkpoint supplied.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from full_policy import ActorCritic,encode,FORMAT

def gae(rewards,values,next_values,terminated,ended,gamma=.99,lam=.95):
    advantages=np.zeros(len(rewards),np.float32);acc=0.
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+gamma*next_values[i]*(1-terminated[i])-values[i]
        acc=delta+gamma*lam*(1-ended[i])*acc
        advantages[i]=acc
    return advantages,advantages+np.asarray(values)

def checkpoint(path,model,env,steps,status):
    torch.save(dict(format=FORMAT,model=model.state_dict(),steps=steps,status=status,
                    action_low=torch.from_numpy(env.low),action_high=torch.from_numpy(env.high),
                    scene='weightless_dynamic_ball_no_table',control_steps=env.skip,
                    full_task_validated=False),path)

def warm_start(model,env,path,device,epochs):
    with np.load(path,allow_pickle=False) as z:
        x,y,t=z['obs'],z['action'],z['time']
    if x.ndim!=2 or x.shape[1]!=63 or y.shape!=(len(x),17) or t.shape!=(len(x),):
        raise ValueError('Need the uploaded 63-sensor/17-action full trace')
    # Use only the moving approach section, not the prolonged failed stall.
    keep=t<=13.3;x=x[keep];y=y[keep]
    if len(x)<100 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Insufficient or nonfinite warm-start data')
    # Trace columns: ball xyz=48:51; original EE target xyz=54:57.
    # Use the teacher's actual reach target to warm-start navigation.
    teacher_goal=x[:,54:57].copy()
    xx=encode(np.concatenate([x,teacher_goal],1))
    yy=np.clip((y-env.mid)/env.half,-.999,.999)
    xx=torch.as_tensor(xx,device=device);yy=torch.as_tensor(yy,device=device)
    opt=torch.optim.Adam(model.actor.parameters(),lr=3e-4)
    for epoch in range(epochs):
        ids=torch.randperm(len(x),device=device);losses=[]
        for batch in ids.split(256):
            loss=(model.action(xx[batch])-yy[batch]).square().mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        if epoch%10==0 or epoch==epochs-1:
            print(f'BC approach-only epoch={epoch+1} train_mse={np.mean(losses):.6f}',flush=True)
    print('Warm start only: no successful grasp or stopping examples were supplied.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warmstart_trace',required=True)
    p.add_argument('--out',default='full_policy_training')
    p.add_argument('--steps',type=int,default=1000000,help='Policy decisions, each 20 physics steps')
    p.add_argument('--bc_epochs',type=int,default=100)
    p.add_argument('--rollout',type=int,default=1024)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--gui',action='store_true')
    a=p.parse_args()
    if a.steps<1 or a.rollout<2 or a.bc_epochs<1:p.error('Positive steps/epochs and rollout >=2 required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(1)
    from robotnik_env import RobotnikEnv
    env=RobotnikEnv(headless=not a.gui,seed=a.seed)
    m=ActorCritic().to(a.device)
    try:
        warm_start(m,env,a.warmstart_trace,a.device,a.bc_epochs)
        checkpoint(out/'approach_warmstart.pt',m,env,0,'warm_start_only')
        opt=torch.optim.Adam(m.parameters(),lr=1e-4)
        state=env.reset();total=0;ep_return=0.;ep=0;successes=0
        with (out/'episodes.jsonl').open('w',buffering=1) as log:
            while total<a.steps:
                states=[];zs=[];logs=[];vals=[];rews=[];nvals=[];terms=[];ends=[];rollout_episodes=[]
                for _ in range(min(a.rollout,a.steps-total)):
                    s=torch.as_tensor(encode(state),device=a.device)
                    with torch.no_grad():
                        dist=m.distribution(s);z=dist.sample();action=torch.tanh(z)
                        # Ratios use latent Normal log probabilities: the fixed tanh
                        # Jacobian cancels between old/new policy at the same sample.
                        lp=dist.log_prob(z).sum();v=m.value(s)
                    ns,r,term,trunc,info=env.step(action.cpu().numpy())
                    with torch.no_grad():
                        nv=m.value(torch.as_tensor(encode(ns),device=a.device)).item()
                    states.append(s.cpu().numpy());zs.append(z.cpu().numpy());logs.append(lp.item())
                    vals.append(v.item());rews.append(r);nvals.append(nv);terms.append(float(term));ends.append(float(term or trunc))
                    state=ns;total+=1;ep_return+=r
                    if term or trunc:
                        ep+=1;successes+=int(info['success'])
                        record=dict(episode=ep,steps=total,return_=ep_return,
                                    candidate_success=info['success'],safety_stop=info['safety_stop'],
                                    ee_ball_distance=info['ee_ball_distance'],
                                    ball_goal_distance=info['ball_goal_distance'],
                                    safety_reasons=info['safety_reasons'],
                                    arm_joint_speeds=np.round(info['joint_velocities'],3).tolist())
                        log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                        rollout_episodes.append(record);ep_return=0.;state=env.reset()
                if (len(rollout_episodes)>=5 and
                    sum(e['safety_stop'] for e in rollout_episodes)/len(rollout_episodes)>=.8 and
                    not any(e['candidate_success'] for e in rollout_episodes)):
                    raise RuntimeError('STOPPED: >=80% of completed rollout episodes hit safety limits '
                                       'with zero successes. Inspect episodes.jsonl safety_reasons; '
                                       'no PPO update was run on this failed rollout.')
                adv,ret=gae(rews,vals,nvals,terms,ends)
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                ss=torch.as_tensor(np.stack(states),device=a.device)
                zz=torch.as_tensor(np.stack(zs),device=a.device)
                old=torch.as_tensor(logs,device=a.device)
                aa=torch.as_tensor(adv,device=a.device);rr=torch.as_tensor(ret,device=a.device)
                for epoch in range(6):
                    for ids in torch.randperm(len(ss),device=a.device).split(256):
                        d=m.distribution(ss[ids]);lp=d.log_prob(zz[ids]).sum(-1)
                        ratio=torch.exp(lp-old[ids]);surrogate=torch.minimum(ratio*aa[ids],ratio.clamp(.8,1.2)*aa[ids])
                        value_loss=(m.value(ss[ids])-rr[ids]).square().mean()
                        # Entropy is that of the latent Gaussian, an exploration surrogate.
                        loss=-surrogate.mean()+.5*value_loss-.001*d.entropy().sum(-1).mean()
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
                checkpoint(out/'latest.pt',m,env,total,'ppo_unvalidated')
                print(f'CHECKPOINT steps={total} episodes={ep} candidate_successes={successes}',flush=True)
        checkpoint(out/'final.pt',m,env,total,'ppo_unvalidated')
    finally:
        env.close()
if __name__=='__main__':main()
