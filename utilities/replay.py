from collections import deque

import torch
import numpy as np

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)

    def push(self, values):
        self._buffer.append(values)

    def __iter__(self):
        return iter(self._buffer)
    
    def get_data(self):
        x_array, y_array = [], []

        for x, y in self:
            x_array.append(x)
            y_array.append(y)

        x_array = np.concatenate(x_array)
        y_array = np.concatenate(y_array)
        return torch.from_numpy(x_array).float(), torch.from_numpy(y_array).float()