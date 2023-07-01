import optuna 
import torch
from torch.utils.data import DataLoader, Subset

import numpy as np

from utilities import Tuner, constants as cnst
from utilities.backbones import EarlyStopper, NN


class Strategy:

    #DEBUG
    epochs=500

    model = None
    model_class = NN
    model_configs = {
        "last_activation": "Softmax",
        "criterion": "CrossEntropyLoss", 
        "optimizer": "SGD",
        "early_stop": True
    }
    
    def __init__(self, 
                 data,
                 idx_lb,
                 random_seed,
                 model_arch_name="MLP"):
        
        # Reproducibility
        torch.manual_seed(random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_configs = data["train"].configs
        self.model_configs.update({"metrics_dict":self.data_configs["upstream_metrics"], 
                                   "batch_size":self.data_configs["batch_size"]})


        self.data = data
        self.train_dataset = data['train']

        for key_name in ["val", "test"]:
            if key_name in self.data:
                setattr(self, f"{key_name}_loader", DataLoader(data[key_name], 
                                                               batch_size=self.data_configs["batch_size"]))

        self.idx_intact = np.arange(self.train_dataset.y.shape[0])
        self.idx_lb = idx_lb
        
        self.model_arch_name = model_arch_name
        self.tuner = Tuner(parent=self, random_seed=random_seed)
        self.random_seed = random_seed


    @property
    def idx_ulb(self):
        return np.delete(self.idx_intact, self.idx_lb)
    
    @property
    def train_loader(self):
        return DataLoader(Subset(self.train_dataset, self.idx_lb), 
                          batch_size=self.data_configs["batch_size"], 
                          shuffle=True, 
                          drop_last=True)

    def reset_model(self):
        for seq in self.model.children():
            for layer in seq.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
    
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
                if not self.model:
                   self.model = self.initialize_model()  # to make sure that we initialize the model
                return func(self, *args, **kwargs)
        return wrapper

    def get_scores(self):
        pass

    def query(self):
        return self.idx_ulb[self.get_scores().argmax()]

    def add_new_inst(self, idx):
        self.idx_lb = np.append(self.idx_lb, idx)
        assert len(self.idx_ulb)
    
    def get_labeled(self):
        return self.train_dataset[self.idx_lb]
    
    def get_unlabeled(self):
        return self.train_dataset[self.idx_ulb]

    def initialize_model(self):
        return self.model_class(self.device, **self.model_configs)
    
    def update_model_configs(self, new_configs):
        self.model_configs.update(new_configs)
        self.model = self.initialize_model()
    
    def eval_model(self, split_name):
        total_loss = 0
        loader = getattr(self, f"{split_name}_loader")
        with torch.no_grad():
            for inputs, targets in loader:

                targets = targets.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.model(inputs)

                batch_loss = self.model.criterion(predictions, targets)
                total_loss += batch_loss.item()
                self.model.metrics_set.update(inputs=predictions, targets=targets)  

        return total_loss, self.model.metrics_set.flush()

    @initilize_first
    def train_model(self, trial=None):

        # TO DISABLE DROPOUT (and Normalization if it is added)
        self.model.eval()
        
        early_stopper = EarlyStopper()

        for epoch_num in range(self.epochs):
            train_loss = 0

            for inputs, targets in self.train_loader:

                targets = targets.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.model(inputs.float())
                
                batch_loss = self.model.criterion(predictions, targets.float())
                train_loss += batch_loss.item()
                self.model.metrics_set.update(inputs=predictions, targets=targets)
                self.model.zero_grad()
                batch_loss.backward()
                self.model.optimizer.step()
                                 
            train_metrics = self.model.metrics_set.flush()
            val_loss, val_metrics = self.eval_model("val")

            if trial:
                trial.report(val_loss, epoch_num)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
            if early_stopper.early_stop(val_loss):
                break