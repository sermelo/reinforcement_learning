import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lib.actor import Actor
from lib.critic import Critic
from lib.memory import Memory

class Agent(object):
    def __init__(self, env, batch_size):
        self.batch_size = batch_size
        self.tau = 1e-2
        memory_size = 50000
        self.gamma = 0.99
        actor_learning_rate=1e-4
        critic_learning_rate=1e-3
        self.critic_loss_fn  = nn.MSELoss()

        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, env.action_space.low)
        self.actor_target = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, env.action_space.low)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic = Critic(env.observation_space.shape[0] + env.action_space.shape[0])
        self.critic_target = Critic(env.observation_space.shape[0] + env.action_space.shape[0])
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        self.memory = Memory(memory_size)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        tensor_state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        tensor_action = self.actor.noisy_forward(tensor_state)
        #tensor_action = self.actor.forward(tensor_state)
        return tensor_action.detach().numpy()[0]

    def get_test_action(self, state):
        tensor_state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        tensor_action = self.actor.forward(tensor_state)
        return tensor_action.detach().numpy()[0]

    def save(self, state, action, reward, new_state, fail):
        self.memory.push(state, action, reward, new_state, fail)

    def update(self):
        if (len(self.memory) < self.batch_size):
            return
        states, actions, rewards, next_states, fails = self.memory.get_batch(self.batch_size)

        states_q_values = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_states_q_value = self.critic_target.forward(next_states, next_actions.detach())
        not_fails = (fails == 0).view(fails.size()[0], 1)
        next_states_q_value = next_states_q_value * not_fails
        new_q_value = rewards + (self.gamma * next_states_q_value)
        critic_loss = self.critic_loss_fn(states_q_values, new_q_value)

        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.reset_noise()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

