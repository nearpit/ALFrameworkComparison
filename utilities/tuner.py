import copy
import optuna
from torch.utils.data import DataLoader, Subset

import utilities

class BaseClass:
    all_hypers = {"lr", "weight_decay", "layers_size", "drop_rate"}

    def __init__(self, pool, clf, hpo_mode, split, tunable_hypers=None):
        self.pool = pool
        self.clf = clf
        if tunable_hypers is None:
            tunable_hypers = self.all_hypers.copy()
        self.tunable_hypers = tunable_hypers
        self.hpo_mode = hpo_mode
        self.split = split

    def add_input_output_size(self, layers_size):
        layers_size.insert(0, self.pool.n_features)
        layers_size.append(self.pool.n_classes)
        return layers_size
    
class Tuner(BaseClass):
                                              # How many fold it warms up
    study_configs = {}
    share_warmup_steps = 0.4

    #CAVEAT check the objective direction   #DEBUG  
    def __init__(self, n_trials, direction="minimize", n_startup_trials=10, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        # optuna.logging.set_verbosity(optuna.logging.WARNING) #DEBUG
        self.n_trials = n_trials  
        if self.split == "dynamic":
            n_warmup_steps = int((self.pool.dynamic_splits)*self.share_warmup_steps)
        else:
            n_warmup_steps = int((self.clf.epochs)*self.share_warmup_steps)
        self.study_configs["pruner"] = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps, n_startup_trials=n_startup_trials)
        self.study_configs["direction"] = direction
        self.study_configs["sampler"] = optuna.samplers.TPESampler(seed=self.pool.random_seed)


    def __call__(self):
        study = optuna.create_study(**self.study_configs)
        study.optimize(Objective(clf=self.clf, 
                                 pool=self.pool,
                                 hpo_mode=self.hpo_mode,
                                 split = self.split,
                                 tunable_hypers=self.tunable_hypers), 
                        n_trials=self.n_trials)
        return study.best_trial.user_attrs["suggested_dict"], study.best_trial.user_attrs["trained_model"], study.best_trial.user_attrs["train_val_perf"]
    

class Objective(BaseClass):
    ranges = {
        "weight_decay": {"low": 1e-6, "high":1e-1, "log":True},
        "depth": {"low": 1, "high":5},
        "width": {"low": 2, "high": 96},
        "drop_rate": {"low": 1e-3, "high":0.5, "log":True},
        "lr": {"low": 1e-4, "high":5e-1},
        "dim_coef": {"low": 0.1, "high": 0.5}
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "MLP" in self.clf.model_arch_name:
            self.define_func = self.define_MLP
        elif "AE" in self.clf.model_arch_name:
            self.define_func = self.define_AE


    def __call__(self, trial):
        suggest_dict = self.suggest_params(trial)
        trial.set_user_attr("suggested_dict", suggest_dict)
        self.clf.update_model_configs(suggest_dict)
        val_loss = utilities.OnlineAvg()
        if self.split == "dynamic":
            for fold_num, (train_idx, val_idx) in enumerate(self.pool.CV_splits()):
                train_loader, val_loader = self.pool.get_train_val_loaders(train_idx, val_idx)
                train_perf, val_perf = self.clf.fit(train_loader=train_loader, val_loader=val_loader)
                val_loss += float(val_perf[0])
                # print(fold_num, suggest_dict, val_perf)
                trial.report(float(val_loss), fold_num)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("trained_model", None)
            trial.set_user_attr("train_val_perf", (None, None))
        else:
            unviolated_train_idx, unviolated_val_idx = self.pool.one_split()
            train_loader, val_loader = self.pool.get_train_val_loaders(unviolated_train_idx, unviolated_val_idx)
            train_perf, val_perf = self.clf.fit(train_loader=train_loader, val_loader=val_loader, trial=trial)
            val_loss += float(val_perf[0])
            trial.set_user_attr("trained_model", self.clf.model)
            trial.set_user_attr("train_val_perf", (train_perf, val_perf))

        return float(val_loss)


    def suggest_params(self, trial):
        suggest_dict = {}
        for key in self.all_hypers:
            if key in self.tunable_hypers:
                if key == "layers_size":
                    suggest_dict[key] = self.add_input_output_size(self.define_func(trial))
                elif key in ["weight_decay", "lr", "drop_rate"]:
                    suggest_dict[key] = trial.suggest_float(key, **self.ranges[key])
                else:
                    raise Exception("Something is wrong with the tuning process")
            else:
                suggest_dict[key] = trial.suggest_categorical(key, [self.clf.model_configs[key]]) # the given hyper
        return suggest_dict
    
    def define_MLP(self, trial):
        layers_size = []
        depth = trial.suggest_int("depth", **self.ranges["depth"])
        for idx in range(depth):
            layers_size.append(trial.suggest_int(f"width_{idx}", **self.ranges["width"]))
        return layers_size

    def define_AE(self, trial):
        current_width = trial.suggest_int("width_0", **self.ranges["width"])
        bottleneck =  trial.suggest_int("bottleneck", self.ranges["width"]["low"], max(self.ranges["width"]["low"], current_width/2))
        diminishing_coef = trial.suggest_float("dim_coef", **self.ranges["dim_coef"])
        layers_size = [current_width, bottleneck, current_width]

        current_width = int(current_width*diminishing_coef)
        i = 1

        while current_width > bottleneck:
            layers_size.insert(i, current_width)
            layers_size.insert(-i, current_width)
            i += 1
            current_width = int(current_width*diminishing_coef)
        return layers_size
 