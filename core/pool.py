import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import datasets


class Pool:
    k_folds = 4

    def __init__(self, data, whole_dataset=False, **kwargs):
        np.random.seed(42)
        self.__dict__.update(**data["train"].configs)
        self.data = data
        self.idx_intact = np.arange(len(self.data["train"].x))

        if whole_dataset:
            self.n_initial_label = self.budget

        self.idx_lb = np.random.choice(self.idx_intact, size=self.n_initial_label, replace=False)
        self.test_loader = DataLoader(data["test"], batch_size=self.batch_size, shuffle=False)

        self.k_folder = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

    @property
    def labeled_set(self):
        return Subset(self.data['train'], self.idx_lb)
    @property
    def train_folder(self):
        return self.k_folder.split(self.labeled_set)
    
    @property
    def splits_loaders(self):
        train_idx, val_idx = datasets.VectoralDataset.conv_split(self.get_len("labeled"), shares=[1/self.k_folds])
        train_loader = DataLoader(Subset(self.labeled_set, train_idx), shuffle=True, drop_last=self.drop_last)
        val_loader = DataLoader(Subset(self.labeled_set, val_idx), shuffle=False)
        return train_loader, val_loader

    @property
    def drop_last(self):
        # drop last if the number of labeled instances is bigger than the batch_size
        return int(self.get_len("labeled")/self.k_folds) > self.batch_size 

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