import torch
from torch.utils.data import DataLoader, Subset

import numpy as np

from utilities import constants as cnst
from utilities.classes import EarlyStopper


class Strategy:
    def __init__(self, 
                 model,
                 data,
                 idx_lb, 
                 epochs=cnst.EPOCHS):
        
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_dataset = data['train']
        self.val_loader = DataLoader(data["val"], batch_size=model.batch_size)
        self.test_loader = DataLoader(data["test"], batch_size=model.batch_size)

        self.intact_idx = np.arange(self.train_dataset.y.shape[0])
        self.idx_lb = idx_lb

        self.epochs = epochs

        self.g = torch.Generator()
        self.g.manual_seed(cnst.RANDOM_STATE)
    
    # def _reset_model(func):
    #     def reset_model(self):
    #         for seq in self.model.children():
    #             for layer in seq.modules():
    #                 if hasattr(layer, 'reset_parameters'):
    #                     layer.reset_parameters()
    #         func(self)
    #     return reset_model

    @property
    def idx_ulb(self):
        return np.delete(self.intact_idx, self.idx_lb)
    
    @property
    def lbl_loader(self):
        return DataLoader(Subset(self.train_dataset, self.idx_lb), batch_size=self.model.batch_size, drop_last=True)
    
    @property
    def ulbl_loader(self):
        return DataLoader(Subset(self.train_dataset, self.idx_ulb), batch_size=self.model.batch_size, drop_last=True)

    def query(self):
        pass

    def update(self, idx):
        self.idx_lb = np.append(self.idx_lb, idx)
        assert len(self.idx_ulb)
    
    def get_labeled(self):
        return self.train_dataset[self.idx_lb]
    
    def get_unlabeled(self):
        return self.train_dataset[self.idx_ulb]

    def eval(self, split_name):
        total_loss = 0
        metric = self.model.metric(device=self.device)
        loader = getattr(self, f"{split_name}_loader")
        with torch.no_grad():
            for inputs, labels in loader:

                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                output = self.model(inputs)

                batch_loss = self.model.criterion(output, labels)
                total_loss += batch_loss.item()
                metric.update(input=output.squeeze(), target=labels.squeeze())
        return total_loss, metric.compute().item()

    # @_reset_model
    def train_clf(self):
        
        early_stopper = EarlyStopper(patience=cnst.PATIENCE, min_delta=cnst.MIN_DELTA)

        for _ in range(self.epochs):
            total_loss_train = 0
            total_loss_val = 0

            train_metric =  self.model.metric(device=self.device)
            val_metric =  self.model.metric(device=self.device)

            for train_input, train_label in self.lbl_loader:

                train_label = train_label.to(self.device)
                train_input = train_input.to(self.device)

                output = self.model(train_input.float())
                
                batch_loss = self.model.criterion(output, train_label)
                total_loss_train += batch_loss.item()

                train_metric.update(input=output.squeeze(), target=train_label.squeeze())
                self.model.zero_grad()
                batch_loss.backward()
                self.model.optimizer.step()
            
                       
            total_acc_train = train_metric.compute()
            loss_val, acc_val = self.eval("val")

            if early_stopper.early_stop(loss_val):
                break
    

