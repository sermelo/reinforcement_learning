import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable

# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()
    self.noise = False
    self.reset_noise()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant_(self.sigma_weight, self.sigma_init)
      init.constant_(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    if self.noise:
        # I think noise should not be varible at all but should not affect because we never learn with noise./print
        x = F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))
    else:
        x = F.linear(input, self.weight, self.bias)
    return x

  def reset_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def noise_on(self):
    noise = True

  def noise_off(self):
    noise = False
