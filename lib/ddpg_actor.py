import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.noisy_linnear import NoisyLinear

class DdpgActor(nn.Module): 
    def __init__(self, input_size, output_size, high, low):
        super(DdpgActor, self).__init__()
        self.hidden_size = 256

        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
 
        self.linear1 = NoisyLinear(input_size, self.hidden_size)
        self.linear2 = NoisyLinear(self.hidden_size, self.hidden_size)
        self.linear3 = NoisyLinear(self.hidden_size, output_size)
 
    def forward(self, state):
        self.__noise_off()
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = x * self.high
        return x

    def noisy_forward(self, state):
        self.__noise_on()
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        x = x * self.high
        return x

    def reset_noise(self):
        sigma = abs(np.random.normal(0, 0.1))
        self.linear1.reset_noise(sigma)
        self.linear2.reset_noise(sigma)
        self.linear3.reset_noise(sigma)

    def __noise_on(self):
        self.linear1.noise_on()
        self.linear2.noise_on()
        self.linear3.noise_on()

    def __noise_off(self):
        self.linear1.noise_off()
        self.linear2.noise_off()
        self.linear3.noise_off()
