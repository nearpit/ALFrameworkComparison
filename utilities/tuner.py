import copy
import optuna
from torch.utils.data import DataLoader, Subset

import utilities

class BaseClass:
    all_hypers = {"lr", "weight_decay", "layers_size"}

    def __init__(self, pool, clf, tunable_hypers=None):
        self.pool = pool
        self.clf = clf
        if tunable_hypers is None:
            tunable_hypers = self.all_hypers.copy()
        self.tunable_hypers = tunable_hypers

    def add_input_output_size(self, layers_size):
        layers_size.insert(0, self.pool.n_features)
        layers_size.append(self.pool.n_classes)
        return layers_size
    
class Tuner(BaseClass):
                                              # How many fold it warms up
    pruner_configs = {"n_startup_trials": 5}
    study_configs = {}
    share_warmup_steps = 0.6

    #CAVEAT check the objective direction   #DEBUG  
    def __init__(self, n_trials, previous_loss=None, direction="minimize", *args, **kwargs):    
        super().__init__(*args, **kwargs)
        optuna.logging.set_verbosity(optuna.logging.WARNING) #DEBUG
        if previous_loss:
            self.callbacks = [StopWhenFoundBetter(previous_loss)]
        else:
            self.callbacks = []
        self.n_trials = n_trials  
        self.study_configs["pruner"] = optuna.pruners.MedianPruner(n_warmup_steps=int((self.pool.n_splits)*self.share_warmup_steps), **self.pruner_configs)
        self.study_configs["direction"] = direction
        self.study_configs["sampler"] = optuna.samplers.TPESampler(seed=self.pool.random_seed)


    def __call__(self):
        study = optuna.create_study(**self.study_configs)
        study.optimize(Objective(clf=self.clf, 
                                 pool=self.pool,
                                 tunable_hypers=self.tunable_hypers), 
                       n_trials=self.n_trials,
                       callbacks=self.callbacks)
        return self.align_params(study.best_params), study.best_trial.user_attrs["avg_val_loss"]
    
    
    def align_params(self, best_params):
        decoded_params = {
            "lr": best_params["lr"],
            "weight_decay": best_params["weight_decay"]
        }

        if "layers_size" in best_params:
            decoded_params["layers_size"] = best_params["layers_size"]
        else:
            width_dict = {k: v for k, v in best_params.items() if "width_" in k}
            layers_size = []
            for idx, _ in enumerate(width_dict.keys()):
                layers_size.append(width_dict[f"width_{idx}"])
            if self.clf.model_arch_name  == "AE":
                # TODO check whether it works correctly
                layers_size.extend(layers_size[1::-1])
            
            decoded_params["layers_size"] = self.add_input_output_size(layers_size)

        return decoded_params

class Objective(BaseClass):
    ranges = {
        "weight_decay": {"low": 1e-6, "high":1e-1, "log":True},
        "depth": {"low": 1, "high":5},
        "width": {"low": 2, "high": 96},
        "lr": {"low": 1e-4, "high":5e-1}
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "MLP" in self.clf.model_arch_name:
            self.define_func = self.define_MLP
        elif "AE" in self.clf.model_arch_name:
            self.define_func = self.define_AE


    def __call__(self, trial):
        suggest_dict = self.suggest_params(trial)
        self.clf.update_model_configs(suggest_dict)
        val_loss = utilities.OnlineAvg()
        for fold_num, (train_idx, val_idx) in enumerate(self.pool.get_unviolated_splitter(tune=True)):
            train_loader, val_loader = self.pool.get_train_val_loaders(train_idx, val_idx)
            train_perf, val_perf = self.clf.fit(train_loader=train_loader, val_loader=val_loader)
            val_loss += float(val_perf[0])
            # print(fold_num, suggest_dict, val_perf)

            trial.report(float(val_loss), fold_num)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("avg_val_loss", float(val_loss))
        return float(val_loss)


    def suggest_params(self, trial):
        suggest_dict = {}
        for key in self.all_hypers:
            if key in self.tunable_hypers:
                if key == "layers_size":
                    suggest_dict[key] = self.add_input_output_size(self.define_func(trial))
                elif key in ["weight_decay", "lr"]:
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

    # def define_AE(self, trial):
    #     current_width = trial.suggest_int("width_0", **self.params_ranges["width"])
    #     bottleneck =  trial.suggest_int("bottleneck", self.params_ranges["width"]["low"], current_width - 1) # -1 to exclude upper bound
    #     layers_size = [current_width, bottleneck, current_width]

    #     current_width = trial.suggest_int(f"width_{1}", bottleneck, current_width - 1) # -1 to exclude upper bound
    #     i = 2

    #     while current_width > bottleneck:
    #         layers_size.insert(i, current_width)
    #         layers_size.insert(-i, current_width)
    #         current_width = trial.suggest_int(f"width_{i}", bottleneck, current_width - 1) # -1 to exclude upper bound
    #         i += 1
    #     return layers_size

class StopWhenFoundBetter:
    def __init__(self, previous_loss):
        self.previous_loss = float(previous_loss)

    def __call__(self, study, trial):
        if study.direction.name == "MINIMIZE":
            if study.best_value < self.previous_loss:
                study.stop()
        elif study.direction.name == "MAXIMIZE":
            if study.best_value > self.previous_loss:
                study.stop()
 