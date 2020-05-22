import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from lib.sac_actor import SacActor
from lib.critic import Critic
from lib.memory import Memory

class SacAgent(object):
    def __init__(self, env, batch_size):
        self.batch_size = batch_size
        self.tau = 1e-2
        memory_size = 50000
        self.gamma = 0.99
        self.q_lr = 3e-4
        self.actor_lr = 3e-4
        self.alpha_lr = 3e-4

        self.update_step = 0
        self.delay_step = 2

        self.action_range = [env.action_space.low, env.action_space.high]

        self.memory = Memory(memory_size)

        # entropy temperature
        self.alpha = 0.2
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)


        self.actor = SacActor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high, env.action_space.low)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.q_net_1 = Critic(env.observation_space.shape[0] + env.action_space.shape[0])
        self.q_net_1_target = Critic(env.observation_space.shape[0] + env.action_space.shape[0])
        for target_param, param in zip(self.q_net_1_target.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_(param.data)
        self.q_net_1_optimizer = optim.Adam(self.q_net_1.parameters(), lr=self.q_lr)

        self.q_net_2 = Critic(env.observation_space.shape[0] + env.action_space.shape[0])
        self.q_net_2_target = Critic(env.observation_space.shape[0] + env.action_space.shape[0])
        for target_param, param in zip(self.q_net_2_target.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_(param.data)
        self.q_net_2_optimizer = optim.Adam(self.q_net_2.parameters(), lr=self.q_lr)


    def get_test_action(self, state):
        # 100% deterministic. It is not always the best option to do it this way
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor.forward(state)
        action = torch.tanh(mean)
        action = action.detach().squeeze(0).numpy()
        return self.rescale_action(action)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, log_pi = self.actor.sample(state)
        action = action.detach().squeeze(0).numpy()
        return self.rescale_action(action)

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0

    def save(self, state, action, reward, new_state, fail):
        self.memory.push(state, action, reward, new_state, fail)

    def update(self, num=1):
        for _ in range(num):
            self.__one_update()

    def __one_update(self):
        if (len(self.memory) < self.batch_size):
            return
        states, actions, rewards, next_states, fails = self.memory.get_batch(self.batch_size)
        not_fails = (fails == 0).view(fails.size()[0], 1)

        next_actions, next_log_pi = self.actor.sample(next_states)

        next_q_1 = self.q_net_1_target(next_states, next_actions)
        next_q_2 = self.q_net_2_target(next_states, next_actions)
        next_q_target = torch.min(next_q_1, next_q_2) - self.alpha * next_log_pi
        expected_q = rewards + not_fails * self.gamma * next_q_target

        curr_q_1 = self.q_net_1.forward(states, actions)
        curr_q_2 = self.q_net_2.forward(states, actions)
        q1_loss = F.mse_loss(curr_q_1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q_2, expected_q.detach())

        self.q_net_1_optimizer.zero_grad()
        q1_loss.backward()
        self.q_net_1_optimizer.step()

        self.q_net_2_optimizer.zero_grad()
        q2_loss.backward()
        self.q_net_2_optimizer.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.actor.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net_1.forward(states, new_actions),
                self.q_net_2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # target networks
            for target_param, param in zip(self.q_net_1_target.parameters(), self.q_net_1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.q_net_2_target.parameters(), self.q_net_2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

