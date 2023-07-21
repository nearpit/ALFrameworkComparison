import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import ShuffleSplit

class Pool:
    n_splits = 10
    const_seed = 2**32 - 1

    def __init__(self, data, args, val_share=None, n_initially_labeled=None, **kwargs):
        self.random_seed = args.random_seed
        self.set_seed(self.random_seed)
        self.__dict__.update(**data["train"].configs)
        self.data = data
        self.args = args
        self.idx_abs = np.arange(len(self.data["train"].x))

        if val_share is None:
            val_share = args.val_share
        self.val_share = val_share

        if n_initially_labeled is None:
            n_initially_labeled = args.n_initially_labeled
        self.n_initially_labeled = n_initially_labeled
        
        self.set_seed(self.random_seed)

        self.idx_unviolated_lb = np.random.choice(self.idx_abs, size=self.n_initially_labeled, replace=False)
        self.idx_new_lb = np.array([], dtype=int)

        self.set_seed(self.random_seed)
        if "test" in data:
            self.test_loader = DataLoader(data["test"], batch_size=self.batch_size, shuffle=False)

        
        self.set_seed(self.random_seed)
        self.shuffle_splitter = ShuffleSplit(n_splits=self.n_splits, 
                                             test_size=args.val_share,
                                             random_state=self.random_seed)

    def __getitem__(self, idx):
        return self.data["train"][idx]  

    @property
    def idx_all_labeled(self):
        return np.append(self.idx_unviolated_lb, self.idx_new_lb)
    
    @property
    def new_lb_dataset(self):
        return Subset(self.data['train'], self.idx_new_lb)
    
    @property
    def unviolated_lb_dataset(self):
        return Subset(self.data['train'], self.idx_unviolated_lb)
    
    @property
    def all_lb_dataset(self):
        return Subset(self.data['train'], self.idx_all_labeled)
    
    @property
    def drop_last(self):
        # drop last if the number of labeled instances is bigger than the batch_size
        return int(self.get_len("unviolated")*(1-self.val_share)) + self.get_len("new_labeled") > self.batch_size 

    @property
    def idx_ulb(self):
        return np.delete(self.idx_abs, self.idx_all_labeled) 
    
    def get_unviolated_splitter(self, tune=True):
        if tune:
            self.set_seed(seed=self.random_seed)
        return self.shuffle_splitter.split(self.unviolated_lb_dataset)
    
    def fill_up(self):
        self.idx_unviolated_lb = np.random.choice(self.idx_abs, size=len(self.idx_abs), replace=False)

    def update_splitter(self, val_share):
        self.shuffle_splitter = ShuffleSplit(n_splits=self.n_splits, 
                                             test_size=val_share,
                                             random_state=self.random_seed)


    def get_train_val_loaders(self, unviolated_train_idx, unviolated_val_idx):
        unviolated_train_ds = Subset(self.unviolated_lb_dataset, unviolated_train_idx)
        unviolated_val_ds = Subset(self.unviolated_lb_dataset, unviolated_val_idx)

        self.set_seed(seed=self.random_seed)
        train_loader = DataLoader(ConcatDataset((unviolated_train_ds, self.new_lb_dataset)),
                                  batch_size=self.batch_size, 
                                  drop_last=self.drop_last,
                                  shuffle=True)
        
        val_loader = DataLoader(unviolated_val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def get_len(self, pool="total"):
        return len(self.get(pool)[0])
    
    def add_new_inst(self, idx):
        assert len(self.idx_ulb)
        self.idx_new_lb = np.append(self.idx_new_lb, idx)

    def get(self, pool):
        if pool == "all_labeled":
            return self[self.idx_all_labeled]
        elif pool == "unviolated":
            return self[self.idx_unviolated_lb] 
        elif pool == "new_labeled":
            return self[self.idx_new_lb] 
        elif pool == "unlabeled":
            return self[self.idx_ulb] 
        elif pool == "total":
            return self[:]
        elif pool == "test":
            return self.data["test"][:]
        else:
            raise NameError("There is no such name in the pool")
        
    def set_seed(self, seed=None):
        if seed is None:
            seed = self.random_seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
