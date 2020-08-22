from collections import deque
import random
import torch

class Memory(object):
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.no_cost_buffer = deque(maxlen=size)
        self.cost_buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, cost, fail):
        self.buffer.append((state, action, [reward], next_state, [cost], [fail]))
        if cost == 0:
            self.no_cost_buffer.append((state, action, [reward], next_state, [cost], [fail]))
        else:
            self.cost_buffer.append((state, action, [reward], next_state, [cost], [fail]))

    def get_batch(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        costs = []
        fails = []

        batch = random.sample(self.buffer, batch_size)

        for item in batch:
            state, action, reward, next_state, cost, fail = item
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            costs.append(cost)
            fails.append(fail)
        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(costs), torch.BoolTensor(fails)

    def get_cost_training_batch(self, batch_size, proportion=0.5):
        states = []
        actions = []
        rewards = []
        next_states = []
        costs = []
        fails = []

        cost_batch_size = int(batch_size * (1 - proportion))
        cost_batch_size = min(cost_batch_size, len(self.cost_buffer))
        no_cost_batch_size = batch_size - cost_batch_size

        batch = random.sample(self.no_cost_buffer, no_cost_batch_size)
        for item in batch:
            state, action, reward, next_state, cost, fail = item
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            costs.append(cost)
            fails.append(fail)

        batch = random.sample(self.cost_buffer, cost_batch_size)
        for item in batch:
            state, action, reward, next_state, cost, fail = item
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            costs.append(cost)
            fails.append(fail)

        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(costs), torch.BoolTensor(fails)

    def __len__(self):
        return len(self.buffer)

    def cost_buffer_size(self):
        return len(self.cost_buffer)
