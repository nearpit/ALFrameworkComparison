import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, layers_size, last_layer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential()
        for idx, _ in enumerate(layers_size[:-1]):
            self.layers.add_module(f"dense_{idx}", nn.Linear(layers_size[idx], layers_size[idx + 1]))
            if idx < len(layers_size) - 2: # to avoid adding extra activation in the output layer
                self.layers.add_module(f"activation_{idx}", nn.ReLU())
        self.layers.add_module("last_activation", getattr(nn, last_layer)())

    def forward(self, x):
        return self.layers(x).view(-1, 1)