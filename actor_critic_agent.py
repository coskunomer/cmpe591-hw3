from actor_critic_model import ActorCritic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

gamma = 0.99

class ActorCriticAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.state_values = []

    def decide_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        state = state.to(self.model.parameters().__next__().device) 
        
        action_probs, state_value = self.model(state)
        
        action_probs = F.softmax(action_probs, dim=-1)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        
        return action.item()

    def update_model(self):
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(self.rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        self.state_values = torch.tensor(self.state_values, dtype=torch.float32)
        
        policy_loss = 0
        for log_prob, discounted_reward in zip(self.log_probs, discounted_rewards):
            policy_loss += -log_prob * discounted_reward

        critic_loss = F.mse_loss(self.state_values, discounted_rewards)
        total_loss = policy_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.state_values = []


    def add_reward(self, reward):
        self.rewards.append(reward)
