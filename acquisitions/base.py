import torch
from torch.utils.data import DataLoader, Subset

import numpy as np

from utilities import constants as cnst
from utilities.classes import EarlyStopper


class Strategy:
    def __init__(self, 
                 model_arch,
                 model_configs,
                 data,
                 idx_lb, 
                 epochs=cnst.EPOCHS):
        
        # Reproducibility
        torch.manual_seed(cnst.RANDOM_STATE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_arch = model_arch
        self.model_configs = model_configs
        self.clf = self.initialize_model().to(self.device)

        self.data = data
        self.train_dataset = data['train']
        self.test_dataset = data['test']

        self.val_loader = DataLoader(data["val"], batch_size=self.clf.batch_size)
        self.test_loader = DataLoader(data["test"], batch_size=self.clf.batch_size)

        self.intact_idx = np.arange(self.train_dataset.y.shape[0])
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

    @property
    def idx_ulb(self):
        return np.delete(self.intact_idx, self.idx_lb)
    
    @property
    def lb_loader(self):
        return DataLoader(Subset(self.train_dataset, self.idx_lb), 
                          batch_size=self.clf.batch_size, 
                          shuffle=True, 
                          drop_last=True)
    
    @property
    def ulb_loader(self):
        return DataLoader(Subset(self.train_dataset, self.idx_ulb), 
                          batch_size=self.clf.batch_size, 
                          shuffle=True, 
                          drop_last=True)

    def query(self):
        pass

    def update(self, idx):
        self.idx_lb = np.append(self.idx_lb, idx)
        assert len(self.idx_ulb)
    
    def get_labeled(self):
        return self.train_dataset[self.idx_lb]
    
    def get_unlabeled(self):
        return self.train_dataset[self.idx_ulb]

    def initialize_model(self):
        return self.model_arch(**self.model_configs)

    def eval(self, split_name):

        total_loss = 0
        metric = self.clf.metric(device=self.device)
        loader = getattr(self, f"{split_name}_loader")
        with torch.no_grad():
            for inputs, labels in loader:

                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                output = self.clf(inputs)

                batch_loss = self.clf.criterion(output, labels)
                total_loss += batch_loss.item()
                metric.update(input=output.ravel(), target=labels.ravel())
        return total_loss, metric.compute().item()

    def train_clf(self):
        # TO DISABLE DROPOUT (and Normalization if it is added)
        self.clf.eval()
        
        early_stopper = EarlyStopper(patience=cnst.PATIENCE, min_delta=cnst.MIN_DELTA)

        for _ in range(self.epochs):
            total_loss_train = 0
            total_loss_val = 0

            train_metric =  self.clf.metric(device=self.device)
            val_metric =  self.clf.metric(device=self.device)

            for inputs, labels in self.lb_loader:

                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.clf(inputs.float())
                
                batch_loss = self.clf.criterion(predictions, labels)
                total_loss_train += batch_loss.item()

                train_metric.update(input=predictions.ravel(), target=labels.ravel())
                self.clf.zero_grad()
                batch_loss.backward()
                self.clf.optimizer.step()
            
                       
            total_acc_train = train_metric.compute()
            loss_val, acc_val = self.eval("val")

            if early_stopper.early_stop(loss_val):
                break
    

