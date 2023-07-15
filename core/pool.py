import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import RepeatedKFold

import datasets


class Pool:
    split_seed = 42
    repeats = 2

    def __init__(self, data, torch_seed, whole_dataset=False, **kwargs):
        self.torch_seed = torch_seed
        self.set_seed(self.split_seed)
        self.__dict__.update(**data["train"].configs)
        self.data = data
        self.idx_intact = np.arange(len(self.data["train"].x))

        if whole_dataset:
            self.n_initial_label = self.budget
        
        self.set_seed(self.split_seed)

        self.idx_lb = np.random.choice(self.idx_intact, size=self.n_initial_label, replace=False)

        self.set_seed(self.split_seed)
        self.train_val_dataset = ConcatDataset([self.labeled_dataset, self.data["val"]])

        self.set_seed(self.split_seed)
        self.test_loader = DataLoader(data["test"], batch_size=self.batch_size, shuffle=False)

        self.set_seed(self.split_seed)
        self.val_loader = DataLoader(data["val"], batch_size=self.batch_size, shuffle=False)
        
        self.set_seed(self.split_seed)
        self.k_folder = RepeatedKFold(n_splits=int(data["val"].x.shape[0]/self.n_initial_label), 
                                      n_repeats=self.repeats, 
                                      random_state=self.split_seed)

    @property
    def labeled_dataset(self):
        return Subset(self.data['train'], self.idx_lb)
       
    @property
    def train_val_kfold(self):
        return self.k_folder.split(self.train_val_dataset)
    
    @property
    def train_loader(self):
        self.set_seed(self.split_seed)
        loader = DataLoader(self.labeled_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)
        return loader

    @property
    def drop_last(self):
        # drop last if the number of labeled instances is bigger than the batch_size
        return self.get_len("labeled") > self.batch_size 

    @property
    def idx_ulb(self):
        return np.delete(self.idx_intact, self.idx_lb)

    def __getitem__(self, idx):
        return self.data["train"][idx]   

    def get_len(self, pool="total"):
        if pool == "labeled":
            return len(self.idx_lb)
        elif pool == "unlabeled":
            return len(self.idx_ulb) 
        else:
            return self.train_size  
        
    def add_new_inst(self, idx):
        self.idx_lb = np.append(self.idx_lb, idx)
        assert len(self.idx_ulb)
    
    def get(self, pool):
        if pool == "labeled":
            return self.data["train"][self.idx_lb]
        elif pool == "unlabeled":
            return self.data["train"][self.idx_ulb] 
        elif pool == "val":
            return self.data["val"][:]
        elif pool == "test":
            return self.data["test"][:]
        
    def set_seed(self, seed=None):
        if seed is None:
            seed = self.torch_seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)