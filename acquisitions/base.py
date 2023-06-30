import optuna 
import torch
from torch.utils.data import DataLoader, Subset

import numpy as np

from utilities import constants as cnst
from utilities.dl_backbones import EarlyStopper


class Strategy:
    
    upstream_model = None

    def __init__(self, 
                 upstream_arch,
                 upstream_configs,
                 data,
                 idx_lb, 
                 epochs=cnst.EPOCHS):
        
        # Reproducibility
        torch.manual_seed(cnst.RANDOM_STATE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upstream_arch = upstream_arch
        self.upstream_configs = upstream_configs

        self.data = data
        self.train_dataset = data['train']
        self.test_dataset = data['test']

        self.val_loader = DataLoader(data["val"], batch_size=upstream_configs["batch_size"])
        self.test_loader = DataLoader(data["test"], batch_size=upstream_configs["batch_size"])

        self.idx_intact = np.arange(self.train_dataset.y.shape[0])
        self.idx_lb = idx_lb

        self.epochs = epochs
    
    @staticmethod     
    def reset_model(func, model_name):
        def _reset_model(self):
            model = getattr(self, model_name) 
            for seq in model.children():
                for layer in seq.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            func(self)
        return _reset_model
    
    @staticmethod
    def hook_penultimate_once(func):
            called = False  
            def wrapper(self, *args, **kwargs):
                nonlocal called
                if not called:
                    called = True
                    self.embedding_hook() # to make sure that we hook once at the right moment
                return func(self, *args, **kwargs)

            return wrapper
    
    def initilize_first(func):
        def wrapper(self, *args, **kwargs):
                if not self.upstream_model:
                    self.upstream_model = self.initialize_upstream()  # to make sure that we initialize the upstream model
                    
                return func(self, *args, **kwargs)
        return wrapper

    @property
    def idx_ulb(self):
        return np.delete(self.idx_intact, self.idx_lb)
    
    @property
    def lb_loader(self):
        return DataLoader(Subset(self.train_dataset, self.idx_lb), 
                          batch_size=self.upstream_configs["batch_size"], 
                          shuffle=True, 
                          drop_last=True)

    def get_scores(self):
        pass

    def query(self):
        return self.idx_ulb[self.get_scores().argmax()]

    def update(self, idx):
        self.idx_lb = np.append(self.idx_lb, idx)
        assert len(self.idx_ulb)
    
    
    def get_labeled(self):
        return self.train_dataset[self.idx_lb]
    
    def get_unlabeled(self):
        return self.train_dataset[self.idx_ulb]

    def initialize_upstream(self):
        return self.upstream_arch(self.device, **self.upstream_configs)
    
    def update_upstream_configs(self, upstream_configs):
        self.upstream_configs.update(upstream_configs)
        self.upstream_model = self.initialize_upstream()
    
    def eval(self, split_name):

        total_loss = 0

        loader = getattr(self, f"{split_name}_loader")
        with torch.no_grad():
            for inputs, targets in loader:

                targets = targets.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.upstream_model(inputs)

                batch_loss = self.upstream_model.criterion(predictions, targets)
                total_loss += batch_loss.item()
                self.upstream_model.metrics_set.update(inputs=predictions, targets=targets)  

        return total_loss, self.upstream_model.metrics_set.flush()

    @initilize_first
    def train_upstream(self, trial=None):
        # TO DISABLE DROPOUT (and Normalization if it is added)
        self.upstream_model.eval()
        
        early_stopper = EarlyStopper(patience=cnst.PATIENCE, min_delta=cnst.MIN_DELTA)

        for epoch_num in range(self.epochs):
            train_loss = 0

            for inputs, targets in self.lb_loader:

                targets = targets.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.upstream_model(inputs.float())
                
                batch_loss = self.upstream_model.criterion(predictions, targets.float())
                train_loss += batch_loss.item()
                self.upstream_model.metrics_set.update(inputs=predictions, targets=targets)
                self.upstream_model.zero_grad()
                batch_loss.backward()
                self.upstream_model.optimizer.step()
                                 
            train_metrics = self.upstream_model.metrics_set.flush()
            val_loss, val_metrics = self.eval("val")

            if trial:
                trial.report(val_loss, epoch_num)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
            if early_stopper.early_stop(val_loss):
                break
