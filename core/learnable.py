import copy
import torch

from utilities import NN, EarlyStopper, Tuner, OnlineAvg

class Learnable:

    #DEBUG
    avg_val_loss = 0
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

    def __call__(self, x):
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
        total_loss = OnlineAvg()
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
    def fit(self, train_loader, val_loader, trial=None):

        self.reset_model()       # To bring the model to the same starting point
        self.model.eval()        # To disable Dropout 
        # early_stopper = EarlyStopper(patience=self.patience, n_warmup_epochs=self.n_warmup_epochs)
        train_loss = OnlineAvg()

        for epoch_num in range(self.epochs):

            for inputs, targets in train_loader:

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
            val_loss, val_metrics = self.eval(val_loader)
            # if early_stopper.early_stop(float(val_loss)):
            #     break
        return (train_loss, train_metrics),  (val_loss, val_metrics)
       
    def reset_model(self):
        self.pool.set_seed()
        for seq in self.model.children():
            for layer in seq.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def tune_model(self, n_trials, online, tunable_hypers=None):
        pool = copy.copy(self.pool)
        if not online:
            pool.fill_up()
            pool.update_splitter(val_share=0.2)
        new_configs, avg_val_loss = Tuner(pool=pool,
                                    clf=self,
                                    n_trials=n_trials,
                                    tunable_hypers=tunable_hypers,
                                    previous_loss=self.avg_val_loss)()
        # Update the avg loss
        if not self.avg_val_loss or avg_val_loss < self.avg_val_loss:
            self.avg_val_loss = float(avg_val_loss)
            
        self.update_model_configs(new_configs)
        return self.train_model()

    def train_model(self):
        unviolated_train_idx, unviolated_val_idx = next(self.pool.get_unviolated_splitter(tune=False))
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