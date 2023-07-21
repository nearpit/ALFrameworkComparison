from collections import deque

import torch
import numpy as np
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline

class ReplayBuffer:

    def __init__(self, capacity):
        self._x = deque(maxlen=capacity)
        self._y = deque(maxlen=capacity)
        self.feature_encoder = FunctionTransformer(lambda x: x)
        self.target_encoder = FunctionTransformer(lambda x: 1/(1 + np.exp(-x)))

    def __len__(self):
        return len(self._x)

    def push(self, x, y):
        self._x.append(x)
        self._y.append(y)
    
    def get_data(self):
        # self.target_encoder = Pipeline([("sqrt", FunctionTransformer(lambda x: np.sqrt(x))), ("minmax", MinMaxScaler())])

        y_transformed = self.target_encoder.fit_transform(np.concatenate(self._y))
        y_transformed = np.append(y_transformed, 1-y_transformed, axis=-1)
        x_transformed = self.feature_encoder.fit_transform(np.concatenate(self._x))
        return torch.from_numpy(x_transformed).float(), torch.from_numpy(y_transformed).float()