from collections import deque

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ReplayBuffer:

    feature_ecoder = MinMaxScaler
    target_encoder = MinMaxScaler

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
            x_transformed = self.feature_ecoder().fit_transform(torch.stack(x))
            y_transformed = self.target_encoder().fit_transform(torch.stack(y))
            x_array.append(x_transformed)
            #CAVEAT check how it works
            y_array.append(np.append(y_transformed, 1 - y_transformed, axis=-1)) # to align to CE

        x_array = np.concatenate(x_array)
        y_array = np.concatenate(y_array)
        return torch.from_numpy(x_array).float(), torch.from_numpy(y_array).float()