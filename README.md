# **Ömer Coşkun | 2024700024**

## **Overview**

The implementation does not explicitly include `train()` and `test()` methods. Instead, the following commands are used for training adn testing:

- **Train REINFORCE model:**
  ```bash
  python train_vpg.py
  ```
- **Train Actor Critic Model:**
  ```bash
  python train_actor_critic.py
  ```


- **Test REINFORCE model:**
  ```bash
  python test_vpg.py
  ```
- **Test Actor Critic Model:**
  ```bash
  python test_actor_critic.py
  ```

---

## **Results**

### REINFORCE

REINFORCE model was trained for 10.000 episodes with 25 steps per episode. However, the training results does not show any improvement. Many configurations were tested with different reward models, different hyperparameters, however, the results were the same. Below is the plot of last run:

### **REINFORCE Performance Graph**
![REINFORCE Graph](vpg/plot.png)

### Actor Critic

REINFORCE model was trained for 10.000 episodes with 30 steps per episode. However, the training results does not show any improvement. Similar to REINFORCE model, many configurations were tested with different reward models, different hyperparameters, however, the results were the same. Below is the plot of last run:

### **REINFORCE Performance Graph**
![Actor Critic Graph](actor_critic/plot.png)
