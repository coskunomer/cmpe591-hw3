import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]):
        super(ActorCritic, self).__init__()
        
        # Shared layers for actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hl[0]),
            nn.ReLU(),
            nn.Linear(hl[0], hl[1]),
            nn.ReLU(),
            nn.Linear(hl[1], act_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(act_dim) - torch.ones(act_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hl[0]),
            nn.ReLU(),
            nn.Linear(hl[0], hl[1]),
            nn.ReLU(),
            nn.Linear(hl[1], 1)
        )

    def forward(self, x):
        mean = self.actor(x)
        mean = torch.tanh(mean)  # keep mean in [-1, 1]

        log_std = torch.clamp(self.log_std, min=-4.0, max=-1) 
        std = log_std.exp()
        value = self.critic(x)
        if (random.random() < 0.05):
            print(mean.data, std.data)
        return mean, std, value

