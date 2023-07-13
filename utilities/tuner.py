from utilities.backbones import EarlyStopper
import optuna

class BaseClass:
    general_hypers = ["lr", "weight_decay", "layers_size"]

    def __init__(self, pool, model):
        self.pool = pool
        self.model = model

    def add_input_output_size(self, layers_size):
        layers_size.insert(0, self.pool.n_features)
        layers_size.append(self.pool.n_classes)
        return layers_size
    
class Tuner(BaseClass):

    pruner_configs = {"n_startup_trials": 10, "n_warmup_steps": 50}
    study_configs = {"pruner": optuna.pruners.MedianPruner(**pruner_configs)}
    seed = 42

    #CAVEAT check the objective direction   #DEBUG  
    def __init__(self, tunable_hypers, direction="minimize", n_trials = 5, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        optuna.logging.set_verbosity(optuna.logging.WARNING) #DEBUG
        self.tunable_hypers = tunable_hypers
        self.n_trials = n_trials  
        self.study_configs["direction"] = direction
        self.study_configs["sampler"] = optuna.samplers.TPESampler(seed=self.seed)


    def __call__(self):
        study = optuna.create_study(**self.study_configs)
        study.optimize(Objective(model=self.model, pool=self.pool, tunable_hypers=self.tunable_hypers), n_trials=self.n_trials)
        return self.align_params(study.best_params)
    
    
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
            if self.model.model_arch_name  == "AE":
                # TODO check whether it works correctly
                layers_size.extend(layers_size[1::-1])
            
            decoded_params["layers_size"] = self.add_input_output_size(layers_size)

        return decoded_params

class Objective(BaseClass):
    suggest_params = {
        "weight_decay": {"low": 0, "high":1e-2},
        "depth": {"low": 1, "high":4},
        "width": {"low": 4, "high": 92},
        "lr": {"low": 1e-6, "high":1e-1}
        }

    def __init__(self, tunable_hypers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tunable_hypers = tunable_hypers
        if "MLP" in self.model.model_arch_name:
            self.define_func = self.define_MLP
        elif "AE" in self.model.model_arch_name:
            self.define_func = self.define_AE

    def __call__(self, trial):
        suggest_dict = {}
        for key in self.general_hypers:
            if key in self.tunable_hypers:
                if key == "layers_size":
                    suggest_dict[key] = self.add_input_output_size(self.define_func(trial))
                elif key in ["weight_decay", "lr"]:
                    suggest_dict[key] = trial.suggest_float(key, **self.suggest_params[key])
                else:
                    raise Exception("Something is wrong with tuning process")
            else:
                suggest_dict[key] = trial.suggest_categorical(key, [self.model.model_configs[key]]) # the given hyper

        # suggest_dict = {
        #     "weight_decay": trial.suggest_float("weight_decay", **self.suggest_params["weight_decay"]) if "weight_decay" in self.tunable_hypers else  trial.suggest_categorical("weight_decay", [self.model.model_configs["weight_decay"]]),
        #     "lr": trial.suggest_float("lr",  **self.suggest_params["lr"]) if "lr" in self.tunable_hypers else trial.suggest_categorical("lr", [self.model.model_configs["lr"]]),
        #     "layers_size": self.add_input_output_size(self.define_func(trial)) if "layers_size" in self.tunable_hypers else trial.suggest_categorical("layers_size", self.model.model_configs["layers_size"])
        # }

        self.model.update_model_configs(suggest_dict)
        self.model.train_model(trial=trial)
        loss, metrics = self.model.eval_model("val")
        return loss

    def define_MLP(self, trial):
        layers_size = []
        depth = trial.suggest_int("depth", **self.suggest_params["depth"])
        for idx in range(depth):
            layers_size.append(trial.suggest_int(f"width_{idx}", **self.suggest_params["width"]))

        return layers_size

    def define_AE(self, trial):
        current_width = trial.suggest_int("width_0", **self.suggest_params["width"])
        bottleneck =  trial.suggest_int("bottleneck", self.suggest_params["width"]["low"], current_width - 1) # -1 to exclude upper bound
        layers_size = [current_width, bottleneck, current_width]

        current_width = trial.suggest_int(f"width_{1}", bottleneck, current_width - 1) # -1 to exclude upper bound
        i = 2

        while current_width > bottleneck:
            layers_size.insert(i, current_width)
            layers_size.insert(-i, current_width)
            current_width = trial.suggest_int(f"width_{i}", bottleneck, current_width - 1) # -1 to exclude upper bound
            i += 1

        return layers_size
   
