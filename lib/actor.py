import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module): 
    def __init__(self, input_size, output_size, output_multiplier): 
        super(Actor, self).__init__() 
        self.hidden_size = 256 
        self.learning_rate = 1e-4 
        self.output_multiplier = output_multiplier 
 
        #learning noise variables 
        self.low = -2 
        self.high = 2 
        self.mu = 0 
        self.sigma = self.high 
        self.min_sigma = 0.01 
        self.sigma_step = 0.0001 
        #self.sigma_step = 0.00005 
 
        self.linear1 = nn.Linear(input_size, self.hidden_size) 
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size) 
        self.linear3 = nn.Linear(self.hidden_size, output_size) 
 
    def forward(self, state): 
        x = F.relu(self.linear1(state)) 
        x = F.relu(self.linear2(x)) 
        x = torch.tanh(self.linear3(x)) 
        x = x * self.output_multiplier 
        return x 

    def noisy_forward(self, state): 
        actions = self.forward(state) 
        noise = np.random.normal(self.mu, self.sigma) 
        noisy_action = actions + noise 
        if (noisy_action > self.high): 
            noisy_action = torch.tensor([[self.high]]) 
        elif (noisy_action < self.low): 
            noisy_action = torch.tensor([[self.low]]) 
        if (self.sigma > self.min_sigma): 
            self.sigma -= self.sigma_step 
        return noisy_action 
