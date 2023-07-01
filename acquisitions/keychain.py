import os
from collections import deque

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import utilities.constants as cnst
from utilities import funcs, NN, EarlyStopper
from acquisitions import Strategy
from datasets import VectoralDataset

class Keychain(Strategy):

    meta_arch = NN

    def __init__(self, buffer_capacity=10, forward_passes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
        self.buffer = ReplayBuffer(buffer_capacity)



    def get_scores(self):
        self.keychain_iteration()
        new_hypers = self.meta_acq.tuner()
        self.meta_acq.update_model_configs(new_hypers)
        
        self.meta_acq.train_model()

        x, y = self.get_unlabeled()
        ulb_probs = self.model(torch.Tensor(x))
        scores = self.meta_acq.model(ulb_probs)[:, 0] # 0 index is for positive, 1 for negative
        return scores 
                
    
    def keychain_iteration(self):
        best_loss, best_metrics = self.eval_model("val")

        model_path = os.getcwd() + "/temp/keychain_model"
        torch.save(self.model.state_dict(), model_path)
        inputs, targets = [], []
        labeled_pool = self.idx_lb.copy()

        for idx, instance in enumerate(labeled_pool):
            self.idx_lb = np.delete(labeled_pool, idx)
            x, y = self.train_dataset[instance]
            self.reset_model()
            self.train_model()
            loss, metrics = self.eval_model("val")
            with torch.no_grad():
                probs = self.model(torch.Tensor(x))
            inputs.append(probs)
            targets.append(torch.Tensor([max(0, loss - best_loss)]))

            self.idx_lb = labeled_pool.copy()


        self.model.load_state_dict(torch.load(model_path))

        self.buffer.push((inputs, targets))

        self.soak_from_buffer()


    def soak_from_buffer(self):
        x, y = self.buffer.get_data()
        train_idx, val_idx = VectoralDataset.conv_split(y.shape[0], shares=[0.8])
        data = {
            "train": ReplayDataset(x[train_idx], y[train_idx], data_configs=self.data_configs),
            "val": ReplayDataset(x[val_idx], y[val_idx], data_configs=self.data_configs)
        }
        
        # Strategy inception ;D
        self.meta_acq = Strategy(data=data, idx_lb=np.arange(len(train_idx)), random_seed=self.random_seed) 

class ReplayDataset(Dataset):

    def __init__(self, x, y, data_configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.configs = data_configs        

    @property
    def dimensions(self):
        return self.x.shape[1], self.y.shape[1]
    
    def __len__(self):
         return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
    