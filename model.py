import torch.nn as nn
import torch.nn.init as init


class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]) -> None:
        super(VPG, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hl)):
            layers.append(nn.Linear(hl[i-1], hl[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hl[-1], act_dim*2))  # act_dim * (1 for mean + 1 for std)
        
        self.model = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.model(x)
    