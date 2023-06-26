import torch
from torch import nn, optim
from torcheval import metrics
from collections import OrderedDict

import utilities.constants as cnst

class MLP(nn.Module):
    def __init__(self,
                 device,
                 layers_size,
                 last_activation,
                 metrics_dict,
                 criterion, 
                 lr, 
                 weight_decay,
                 batch_size, 
                 optimizer,
                 early_stop=True,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        # self.layers = nn.Sequential(OrderedDict([
        #                             ("dropout_0", nn.Dropout(cnst.DROPOUT_RATE)), 
        #                             ("dense_0", nn.LazyLinear(layers_size[0])), 
        #                             ("activation_0", nn.ReLU())]))
        self.layers = nn.Sequential()
        
        for idx, _ in enumerate(layers_size[:-1]):
            self.layers.add_module(f"dropout_{idx}", nn.Dropout(cnst.DROPOUT_RATE))
            self.layers.add_module(f"dense_{idx}", nn.Linear(layers_size[idx], layers_size[idx + 1]))
            if idx < len(layers_size) - 2: # to avoid adding extra activation in the output layer
                self.layers.add_module(f"activation_{idx+1}", nn.ReLU())
        self.layers.add_module("last_activation", getattr(nn, last_activation)(dim=-1))
        self.metrics_dict = metrics_dict
        self.metrics_set = MetricsSet(metrics_dict=metrics_dict, device=device)
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = getattr(nn, criterion)()

        self.early_stop = early_stop
        self.batch_size = batch_size

    def forward(self, x):
        return self.layers(x)
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("Inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class MetricsSet:
    def __init__(self, metrics_dict, device):
        self.raw_metrics_dict = metrics_dict
        self.device = device
        self.result_dict = None

    def update(self, inputs, targets):
        if not self.result_dict:
            self.result_dict = {}
            for name, configs in self.raw_metrics_dict.items():
                attr = getattr(metrics, name)
                self.result_dict[name] = attr(device=self.device, **configs)
                
        for metric in self.result_dict.values():
            metric.update(inputs, targets.argmax(dim=-1))

    def flush(self):
        results = {key:val.compute().item() for key, val in self.result_dict.items()}
        self.result_dict = None
        return results
    