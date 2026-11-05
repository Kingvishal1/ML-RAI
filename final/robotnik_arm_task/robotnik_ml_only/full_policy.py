"""One actor for all 17 motor commands. No phase selector or hand-coded controller."""
import numpy as np
import torch
from torch import nn

FORMAT='robotnik_full_actor_v1'
OBS_DIM=66
ACT_DIM=17
# Same fixed sensor scaling for warm start, training and evaluation.
SCALE=np.asarray([5]*3+[1]*2+[2]*3+[3]*3+[np.pi]*17+[10]*17+
                 [5]*3+[5]*3+[2]*3+[5]*3+[1]*6+[5]*3,dtype=np.float32)

def encode(obs):
    x=np.asarray(obs,dtype=np.float32).copy()
    if x.shape[-1]!=OBS_DIM or not np.isfinite(x).all():
        raise ValueError('Expected 66 finite sensor values')
    x[...,11:15]=(x[...,11:15]+np.pi)%(2*np.pi)-np.pi
    return np.clip(x/SCALE,-10,10)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(OBS_DIM,256),nn.Tanh(),
                                nn.Linear(256,256),nn.Tanh(),nn.Linear(256,ACT_DIM))
        self.critic=nn.Sequential(nn.Linear(OBS_DIM,256),nn.Tanh(),
                                 nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
        self.log_std=nn.Parameter(torch.full((ACT_DIM,),-3.0))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
    def distribution(self,x):
        return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-5,0).exp())
    def value(self,x):
        return self.critic(x).squeeze(-1)
    def action(self,x):
        return torch.tanh(self.actor(x))

def load_actor(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    if ck.get('format')!=FORMAT:
        raise ValueError('Expected full_actor_v1, not a split or grasp checkpoint')
    m=ActorCritic().to(device);m.load_state_dict(ck['model']);m.eval()
    return m,ck
