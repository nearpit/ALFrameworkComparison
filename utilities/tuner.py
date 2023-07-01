import utilities.constants as cnst
from utilities.backbones import EarlyStopper
import optuna

class BaseClass:
    def __init__(self, parent):
        self.parent = parent
        self.inputs, self.outputs = parent.data["train"].dimensions

    def add_input_output_size(self, layers_size):
        layers_size.insert(0, self.inputs)
        layers_size.append(self.outputs)
        return layers_size

class Tuner(BaseClass):

    pruner_configs = {"n_startup_trials": 10, "n_warmup_steps": 100}
    study_configs = {"pruner": optuna.pruners.MedianPruner(**pruner_configs)}

    #CAVEAT check the objective direction    #DEBUG
    def __init__(self, random_seed, direction="minimize", n_trials = 5, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        optuna.logging.set_verbosity(optuna.logging.WARNING) #DEBUG
        self.n_trials = n_trials  
        self.study_configs["direction"] = direction
        self.study_configs["sampler"] = optuna.samplers.TPESampler(seed=random_seed)


        self.model_arch_name = self.parent.model_arch_name

    def __call__(self):
        study = optuna.create_study(**self.study_configs)
        #TODO chech the attributes of study and find "Objective" instance there
        study.optimize(Objective(self.parent), n_trials=self.n_trials)

        return self.align_params(study.best_params)
    
    
    def align_params(self, best_params):
        decoded_params = {
            "lr": best_params["lr"],
            "weight_decay": best_params["weight_decay"]
        }
    
        width_dict = {k: v for k, v in best_params.items() if "width_" in k}
        layers_size = []
        for idx, _ in enumerate(width_dict.keys()):
            layers_size.append(width_dict[f"width_{idx}"])
        if self.model_arch_name == "AE":
            # TODO check whether it works correctly
            layers_size.extend(layers_size[1::-1])
        
        decoded_params["layers_size"] = self.add_input_output_size(layers_size)

        return decoded_params

class Objective(BaseClass):
    suggest_params = {
        "decay": {"low": 1e-7, "high":1e-1, "log":True},
        "depth": {"low": 1, "high":5},
        "width": {"low": 4, "high": 92},
        "lr": {"low": 1e-4, "high":1e-1}
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_func = getattr(self, f"define_{self.parent.model_arch_name}")

    def __call__(self, trial):
        suggest_dict = {
            "weight_decay": trial.suggest_float("weight_decay", **self.suggest_params["decay"]),
            "lr": trial.suggest_float("lr",  **self.suggest_params["lr"]),
            "layers_size": self.add_input_output_size(self.define_func(trial))
        }

        self.parent.update_model_configs(suggest_dict)
        self.parent.train_model(trial=trial)
        loss, metrics = self.parent.eval_model("val")
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
   