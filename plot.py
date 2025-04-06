import numpy as np
import matplotlib.pyplot as plt

rewards = np.load('rews.npy')

def exponential_moving_average(data, alpha=0.1):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema

ema_rewards = exponential_moving_average(rewards, alpha=0.1)

x = np.arange(len(ema_rewards))

slope, intercept = np.polyfit(x, ema_rewards, 1)
linear_fit = slope * x + intercept

plt.figure(figsize=(12, 6))

plt.plot(rewards, label='Rewards per Episode', color='blue')
plt.plot(ema_rewards, label='EMA (alpha=0.1)', linestyle='--', color='red')

plt.title('Rewards per Episode with EMA')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

plt.tight_layout()
plt.show()
