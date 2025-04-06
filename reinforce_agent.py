import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from model import VPG

gamma = 0.99

class Agent():
    def __init__(self):
        # Initialize the model and optimizer
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def decide_action(self, state):
        # Convert state to a PyTorch tensor if it's a NumPy array
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Ensure the tensor is on the same device as the model
        state = state.to(self.model.parameters().__next__().device)  # Move to the same device as the model
        
        # Forward pass through the network
        action_mean, act_std = self.model(state).chunk(2, dim=-1)

        # Check for NaN values
        if torch.isnan(action_mean).any() or torch.isnan(act_std).any():
            print("Warning: NaN detected in action_mean or action_std.")
            action_mean = torch.zeros_like(action_mean)
            act_std = torch.ones_like(act_std)  # Default to reasonable values

        # Apply softplus to the standard deviation to ensure it's positive
        action_std = F.softplus(act_std) + 5e-2  # Increase variance for exploration
        
        # Sample an action from a normal distribution defined by mean and std
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        
        # Store log probability of the taken action
        log_prob = dist.log_prob(action)
        
        # Store the action and its log probability for later use in the update step
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
        return action

    def update_model(self):
        # Compute the total discounted rewards (returns) from the rewards list
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(self.rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        # Convert rewards to a tensor and normalize them
        discounted_rewards = torch.tensor(discounted_rewards)
        
        # Normalize the discounted rewards with handling small number cases
        if discounted_rewards.numel() > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        else:
            discounted_rewards = discounted_rewards - discounted_rewards.mean()
        
        # Compute the policy gradient loss
        policy_loss = 0
        for log_prob, discounted_reward in zip(self.log_probs, discounted_rewards):
            policy_loss += -log_prob * discounted_reward

        # Sum up the policy_loss to get a scalar value
        policy_loss = policy_loss.sum()  # or use .mean() if you prefer to average the loss

        # Perform backpropagation and optimize the model
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear stored data for the next episode
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def add_reward(self, reward):
        self.rewards.append(reward)

