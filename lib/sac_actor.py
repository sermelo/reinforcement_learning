import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SacActor(nn.Module): 
    def __init__(self, input_size, output_size, high, low):
        super(SacActor, self).__init__()
        self.hidden_size = 256
        self.log_std_min=-20
        self.log_std_max=2

        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
 
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.mean_linear = nn.Linear(self.hidden_size, output_size)
        self.log_std_linear = nn.Linear(self.hidden_size, output_size)
 
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        epsilon = 1e-6
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)# * self.high

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi
