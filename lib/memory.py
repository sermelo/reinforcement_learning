from collections import deque
import random
import torch

class Memory(object):
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, cost, fail):
        self.buffer.append((state, action, [reward], next_state, [cost], [fail]))

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

    def __len__(self):
        return len(self.buffer)

