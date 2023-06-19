import torch
from torch import nn, optim
from torcheval import metrics

import utilities.constants as cnst

class MLP(nn.Module):
    def __init__(self,
                 layers_size,
                 last_activation,
                 metric,
                 criterion, 
                 lr, 
                 weight_decay,
                 batch_size, 
                 optimizer=cnst.OPTIMIZER,
                 early_stop=True,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(nn.LazyLinear(layers_size[0]), nn.ReLU())
        for idx, _ in enumerate(layers_size[1:]):
            self.layers.add_module(f"dense_{idx}", nn.Linear(layers_size[idx], layers_size[idx + 1]))
            if idx < len(layers_size) - 2: # to avoid adding extra activation in the output layer
                self.layers.add_module(f"activation_{idx}", nn.ReLU())
        self.layers.add_module("last_activation", getattr(nn, last_activation)())

        self.metric = getattr(metrics, metric)
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = getattr(nn, criterion)()

        self.early_stop = early_stop
        self.batch_size = batch_size

    def forward(self, x):
        return self.layers(x).view(-1, 1)