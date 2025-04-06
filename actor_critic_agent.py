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
        
        # Convert rewards to tensor
        discounted_rewards = torch.tensor(discounted_rewards)
        
        # Compute advantage 
        for i in range(len(discounted_rewards)):
            advantage = discounted_rewards[i] - self.state_values[i]
            
            # Compute the actor loss
            actor_loss = -self.log_probs[i] * advantage
            
            # Compute the critic losss
            critic_loss = F.mse_loss(self.state_values[i], discounted_rewards[i])
            
            # Total loss is the sum of actor and critic losses
            total_loss = actor_loss + critic_loss
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.state_values = []

    def add_reward(self, reward):
        self.rewards.append(reward)
