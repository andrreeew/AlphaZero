import torch
import torch.nn as nn
from torch.nn import functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.size = size
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.policy_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(in_features=4*size*size, out_features=size*size),
            nn.Softmax()
        )

        self.value_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(in_features=2*size*size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Tanh()
        )


        
    def forward(self, X):
        x = self.conv_layer(X)
        policy = self.policy_layer(x).squeeze().reshape(-1, self.size, self.size).squeeze()
        value = self.value_layer(x).squeeze()
        return policy, value

