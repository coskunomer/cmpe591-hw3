from .actor_critic_model import ActorCritic
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

gamma = 0.99

class ActorCriticAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optimizer = torch.optim.Adam([
                {'params': self.model.actor.parameters(), 'lr': 1e-4},
                {'params': self.model.critic.parameters(), 'lr': 1e-4},
                {'params': [self.model.log_std], 'lr': 3e-2} 
            ])

        
        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.state_values = []

    def decide_action(self, state):
        # Convert NumPy state to tensor
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        # Pass through model
        mean, std, state_value = self.model(state)

        # Create Gaussian distribution and sample action
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # Save info for training
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)

        # Return action
        return action

    def update_model(self):
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(self.rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        # Normalize discounted rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Convert state values to tensor
        state_values = torch.stack(self.state_values).view(-1)

        # Compute losses
        policy_loss = 0
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss += -log_prob * reward

        critic_loss = F.mse_loss(state_values, discounted_rewards)

        # Regularize log_std to prevent exploding gradients
        log_std_reg = 1e-3 * (self.model.log_std ** 2).sum()

        total_loss = policy_loss + critic_loss + log_std_reg

        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.state_values = []


    def add_reward(self, reward):
        self.rewards.append(reward)
