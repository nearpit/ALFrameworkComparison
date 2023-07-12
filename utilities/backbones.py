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
                 drop_rate=0.15, # there is no droprate during training - only for BALD within predictions
                 optimizer="SGD",
                 early_stop=True,
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

        self.early_stop = early_stop
        self.batch_size = batch_size

    def forward(self, x):
        return self.layers(x)
    

class EarlyStopper:
                    #DEBUG
    def __init__(self, patience=100, min_delta=0, starts_with=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = starts_with
        self.min_validation_loss = float("Inf")

    def early_stop(self, validation_loss):
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
