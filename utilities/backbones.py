from torch import nn, optim
from torcheval import metrics

class NN(nn.Module):
    def __init__(self,
                 device,
                 layers_size,
                 last_activation,
                 last_activation_configs,
                 metrics_dict,
                 criterion, 
                 lr, 
                 weight_decay,
                 batch_size, 
                 drop_rate,
                 optimizer="SGD",
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential()
        self.layers.add_module(f"dense_0", nn.Linear(layers_size[0], layers_size[1]))
        self.layers.add_module(f"activation_0", nn.ReLU())

        for idx, _ in enumerate(layers_size[1:-1]):
            self.layers.add_module(f"dropout_{idx+1}", nn.Dropout(drop_rate))
            self.layers.add_module(f"dense_{idx+1}", nn.Linear(layers_size[idx+1], layers_size[idx + 2]))
            if idx < len(layers_size) - 3: # to avoid adding extra activation in the output layer
                self.layers.add_module(f"activation_{idx+1}", nn.ReLU())
        self.layers.add_module("last_activation", getattr(nn, last_activation)(**last_activation_configs))
        self.metrics_dict = metrics_dict
        self.metrics_set = MetricsSet(metrics_dict=metrics_dict, device=device)
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = getattr(nn, criterion)()

        self.batch_size = batch_size

    def forward(self, x):
        return self.layers(x)
    

class EarlyStopper:
                    #DEBUG
    def __init__(self, patience, min_delta=0, n_warmup_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.idx = -1
        self.counter = 0
        self.n_warmup_epochs = n_warmup_epochs
        self.min_validation_loss = float("Inf")

    def early_stop(self, validation_loss):
        self.idx += 1

        if self.idx > self.n_warmup_epochs:
            if validation_loss <= self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    self.counter = 0
                    self.min_validation_loss = float("Inf")
                    return True
        return False
    
class MetricsSet:
    def __init__(self, metrics_dict, device):
        self.raw_metrics_dict = metrics_dict
        self.device = device
        self.result_dict = {}
        for name, configs in self.raw_metrics_dict.items():
            attr = getattr(metrics, name)
            self.result_dict[name] = attr(device=self.device, **configs)

    def update(self, inputs, targets):
        for metric in self.result_dict.values():
            metric.update(inputs, targets.argmax(dim=-1))

    def flush(self):
        results = {key:val.compute().item() for key, val in self.result_dict.items()}

        self.result_dict = {key:val.reset() for key, val in self.result_dict.items()} #reset values
        return results
