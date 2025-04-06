import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

gamma = 0.99  # Discount factor

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]):
        super(ActorCritic, self).__init__()
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hl[0]),
            nn.ReLU(),
            nn.Linear(hl[0], hl[1]),
            nn.ReLU(),
            nn.Linear(hl[1], act_dim)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hl[0]),
            nn.ReLU(),
            nn.Linear(hl[0], hl[1]),
            nn.ReLU(),
            nn.Linear(hl[1], 1)  
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
