import torch
import optuna
from torch import nn

from utilities import NN, EarlyStopper, Tuner

class Learnable:

    #DEBUG
    epochs=1000

    model = None
    model_class = NN
    model_configs = {"MLP_clf": {"last_activation": "Softmax",
                                 "last_activation_configs": {"dim":-1},
                                 "criterion": "CrossEntropyLoss"},
                     "MLP_reg": {"last_activation": "Identity",
                                 "last_activation_configs": {},
                                 "criterion": "MSELoss"},
                     "AE": {"last_activation": "Identity",
                            "last_activation_configs": {},
                            "criterion": "MSELoss"}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, 
                 pool,
                 random_seed,
                 model_configs=None,
                 model_arch_name="MLP_clf"):
        
        self.random_seed = random_seed
        # Reproducibility
        torch.manual_seed(random_seed)
        self.pool = pool
        self.model_configs = self.model_configs[model_arch_name].copy()

        if model_configs is None:
            self.model_configs.update({"metrics_dict":pool.metrics_dict,
                                    "batch_size":pool.batch_size})
        else:
            self.model_configs.update(model_configs)
            self.train_model()

        self.model_arch_name = model_arch_name

    def __call__(self, x):
      with torch.no_grad():
        return self.model(x.to(self.device))

    @staticmethod
    def hook_once(func):
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
    
    def initialize_model(self):
        return self.model_class(self.device, **self.model_configs)
    
    def update_model_configs(self, new_configs, seed=None):
        self.model_configs.update(new_configs)
        if seed is None:
            seed = self.random_seed
        torch.manual_seed(seed)
        self.model = self.initialize_model()
        self.model.to(self.device)
    
    def eval_model(self, split_name):
        total_loss = 0
        loader = getattr(self.pool, f"{split_name}_loader")
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

        self.model.eval()        # To disable Dropout 
        
        early_stopper = EarlyStopper()
        for epoch_num in range(self.epochs):
            train_loss = 0

            for inputs, targets in self.pool.train_loader:

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

    def reset_model(self, seed=None):
        if seed is None:
            seed = self.random_seed
        torch.manual_seed(seed)
        for seq in self.model.children():
            for layer in seq.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
    def tune_model(self):
        self.update_model_configs(Tuner(pool=self.pool, model=self)())
        self.train_model()
