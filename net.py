import torch
import torch.nn as nn
from torch.nn import functional as F

class Residual(nn.Module): 
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



class PolicyValueNet(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.size = size

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.residual_layer = nn.Sequential(
            Residual(128, 128),
            Residual(128, 128)
        )

        self.policy_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(in_features=2*size*size, out_features=size*size),
            nn.Softmax()
        )


        self.value_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(in_features=size*size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Tanh()
        )

        
    def forward(self, X):
        x = self.residual_layer(self.conv_layer(X))
        policy = self.policy_layer(x).squeeze().reshape(-1, self.size, self.size).squeeze()
        value = self.value_layer(x).squeeze()
        return policy, value

