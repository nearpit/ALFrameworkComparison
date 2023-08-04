import copy, torch, optuna

from utilities import NN, Tuner, OnlineAvg

class Learnable:

    #DEBUG
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
                 model_arch_name="MLP_clf",
                 n_warmup_epochs=100,
                 patience=20,
                 epochs=200):
        
        self.random_seed = random_seed
        
        self.n_warmup_epochs = n_warmup_epochs
        self.epochs = epochs
        self.patience = patience

        self.pool = pool
        self.model_arch_name = model_arch_name
        self.model_configs = self.model_configs[model_arch_name].copy()


        if model_configs is None:
            self.model_configs.update({"metrics_dict":pool.metrics_dict,
                                       "batch_size":pool.batch_size})
        else:
            self.update_model_configs(model_configs)

    def __call__(self, x, mc_dropout=False):
      if mc_dropout:
          self.model.train()
      else:
          self.model.eval()

      with torch.no_grad():
        return self.model(x.to(self.device))
    
    def initilize_first(func):
        def wrapper(self, *args, **kwargs):
                if not self.model:
                   self.model = self.initialize_model()  # to make sure that we initialize the model
                return func(self, *args, **kwargs)
        return wrapper
    
    def initialize_model(self):
        self.pool.set_seed()
        return self.model_class(self.device, **self.model_configs)
    
    def update_model_configs(self, new_configs):
        self.model_configs.update(new_configs)
        self.model = self.initialize_model()
        self.model.to(self.device)
        self.embedding_hook()
    
    def eval(self, loader):
        self.model.eval()
        total_loss = OnlineAvg()

        with torch.no_grad():
            for inputs, targets in loader:

                targets = targets.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self(inputs.float())

                batch_loss = self.model.criterion(predictions, targets)
                total_loss += batch_loss.item()
                self.model.metrics_set.update(inputs=predictions, targets=targets)  

        return total_loss, self.model.metrics_set.flush()

    @initilize_first
    def fit(self, train_loader, val_loader, trial=None):
        self.model.train()
        self.reset_model()       # To bring the model to the same starting point
        train_loss = OnlineAvg()

        for epoch_num in range(self.epochs):

            for inputs, targets in train_loader:

                targets = targets.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.model(inputs.float())
                
                batch_loss = self.model.criterion(predictions, targets.float())
                train_loss += batch_loss.item()
                self.model.zero_grad()
                batch_loss.backward()
                self.model.optimizer.step()

            train_loss, train_metrics = self.eval(loader=train_loader)
            val_loss, val_metrics = self.eval(val_loader)
            
            if trial:
                trial.report(float(val_loss), epoch_num)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return (train_loss, train_metrics),  (val_loss, val_metrics)
       
    def reset_model(self):
        self.pool.set_seed()
        for seq in self.model.children():
            for layer in seq.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def tune_model(self, n_trials, hpo_mode, split):
        pool = copy.copy(self.pool)
        if hpo_mode=="constant" and split == "whole":
            pool.fill_up()
        new_configs, new_model, (train_perf, val_perf) = Tuner(pool=pool,
                                                               clf=self,
                                                               n_trials=n_trials,
                                                               hpo_mode=hpo_mode,
                                                               split=split)()
        if new_model is None: #dynamic split
            self.update_model_configs(new_configs)
            return self.train_model()
        else: #static split
            self.model = new_model
            test_perf = None
            if hasattr(self.pool, "test_loader"):
                test_perf = self.eval(loader=self.pool.test_loader)
            return train_perf, val_perf, test_perf

    def train_model(self):
        unviolated_train_idx, unviolated_val_idx = self.pool.one_split()
        train_loader, val_loader = self.pool.get_train_val_loaders(unviolated_train_idx, unviolated_val_idx)
        train_perf, val_perf = self.fit(train_loader=train_loader, val_loader=val_loader)
        test_perf = None
        if hasattr(self.pool, "test_loader"):
            test_perf = self.eval(loader=self.pool.test_loader)
        return train_perf, val_perf, test_perf
    

    def embedding_hook(self):
        # penultimate layer hook
        total_layer_depth = len(self.model_configs["layers_size"])
        penultimate_layer_name = f"dense_{total_layer_depth - 2}" 
        penultimate_layer = getattr(self.model.layers, penultimate_layer_name)
        penultimate_layer.register_forward_hook(self.get_activation(penultimate_layer_name))

    # auxiliary function for latent representations
    def get_activation(self, name):
        def hook(model, input, output):
            value = torch.clone(output.detach())
            self.latent = value
        return hook