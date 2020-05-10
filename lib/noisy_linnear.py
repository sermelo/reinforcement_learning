import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable

# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
    self.sigma_bias = Parameter(torch.Tensor(out_features))
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()
    self.noise = False
    self.reset_noise()

  def reset_parameters(self):
    init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
    init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))

  def forward(self, input):
    if self.noise:
        x = F.linear(input, self.weight + self.epsilon_weight, self.bias + self.epsilon_bias)
    else:
        x = F.linear(input, self.weight, self.bias)
    return x

  def reset_noise(self, sigma=0.017):
    self.epsilon_weight = torch.FloatTensor(self.out_features, self.in_features).normal_(0, sigma)
    self.epsilon_bias = torch.FloatTensor(self.out_features).normal_(0, sigma)

  def noise_on(self):
    self.noise = True

  def noise_off(self):
    self.noise = False
