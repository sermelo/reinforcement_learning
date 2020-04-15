from collections import deque
import random
import torch

class Memory(object):
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, [reward], next_state, done))

    def get_batch(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        batch = random.sample(self.buffer, batch_size)

        for item in batch:
            state, action, reward, next_state, done = item
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(dones)

    def __len__(self):
        return len(self.buffer)

