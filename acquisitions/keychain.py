import os
import torch
from collections import deque
import numpy as np

from acquisitions import Strategy

class Keychain(Strategy):
    def __init__(self, capacity=10, forward_passes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
        self.buffer = ReplayBuffer(capacity)
    
    def get_scores(self):
        scores = [1, 0]
        self.collect_new_keys_experience()
        self.train_key_agent()

            
        return scores
    
    def train_key_agent(self):
        data = [x for x in self.buffer]
    
    def collect_new_keys_experience(self):
        best_loss, best_metrics = self.eval("val")

        model_path = os.getcwd() + "/temp/keychain_model"
        torch.save(self.upstream_model.state_dict(), model_path)
        inputs, targets = [], []
        labeled_pool = self.idx_lb.copy()

        for idx, instance in enumerate(labeled_pool):
            self.idx_lb = np.delete(labeled_pool, idx)
            x, y = self.train_dataset[instance]
            self.upstream_model = self.initialize_upstream()
            self.train_upstream()
            loss, metrics = self.eval("val")
            with torch.no_grad():
                probs = self.upstream_model(torch.Tensor(x))
            inputs.append(torch.Tensor([probs]))
            targets.append(torch.Tensor([max(0, loss - best_loss)]))

            self.idx_lb = labeled_pool.copy()


        self.upstream_model.load_state_dict(torch.load(model_path))

        self.buffer.push((inputs, targets))

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
        return torch.cat(x_array), torch.cat(y_array)
    