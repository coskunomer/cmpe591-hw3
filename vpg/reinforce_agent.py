import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from .model import VPG

gamma = 0.99

class Agent():
    def __init__(self):
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def decide_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        state = state.to(self.model.parameters().__next__().device) 
        
        action_mean, act_std = self.model(state).chunk(2, dim=-1)

        if torch.isnan(action_mean).any() or torch.isnan(act_std).any():
            print("Warning: NaN detected in action_mean or action_std.")
            action_mean = torch.zeros_like(action_mean)
            act_std = torch.ones_like(act_std) 

        action_std = F.softplus(act_std) + 5e-2 
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
        return action

    def update_model(self):
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(self.rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        discounted_rewards = torch.tensor(discounted_rewards)
        
        if discounted_rewards.numel() > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        else:
            discounted_rewards = discounted_rewards - discounted_rewards.mean()
        
        policy_loss = 0
        for log_prob, discounted_reward in zip(self.log_probs, discounted_rewards):
            policy_loss += -log_prob * discounted_reward

        policy_loss = policy_loss.sum()  

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.actions = []
        self.log_probs = []

    def add_reward(self, reward):
        self.rewards.append(reward)

