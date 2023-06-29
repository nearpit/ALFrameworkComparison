import json
import optuna
import torch

import numpy as np 
from argparse import ArgumentParser
from optuna.trial import TrialState

import utilities.constants as cnst
from utilities.dl_backbones import EarlyStopper, MLP
from acquisitions import Strategy

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", help="What dataset to train on", required=True, choices=["dna", "splice", "toy"])
    parser.add_argument("-a", "--algorithm", help="What active learning algorithm to evaluate", choices=["random", 
                                                                                                         "cheating",
                                                                                                         "bald", 
                                                                                                         "coreset",
                                                                                                         "entropy"])
    return parser.parse_args()


def define_MLP(trial, data):
    input_size, output_size = data["train"].dimensions
    depth = trial.suggest_int("depth", cnst.DEPTH_RANGE[0], cnst.DEPTH_RANGE[1])
    layers_size = [input_size]
    for idx in range(depth):
        layers_size.append(trial.suggest_int(f"width_{idx}", cnst.WIDHT_RANGE[0], cnst.WIDHT_RANGE[1]))
    layers_size.append(output_size)

    return layers_size


def define_AE(trial, data):
    input_size, output_size = data.dimensions
    current_width = trial.suggest_int("width_0", cnst.WIDHT_RANGE[0], cnst.WIDHT_RANGE[1])
    bottleneck =  trial.suggest_int("bottleneck", cnst.WIDHT_RANGE[0] - 1, current_width - 1) # -1 to exclude upper bound
    layers_size = [current_width, bottleneck, current_width]

    current_width = trial.suggest_int(f"width_{1}", bottleneck, current_width - 1) # -1 to exclude upper bound
    i = 2

    while current_width > bottleneck:
        layers_size.insert(i, current_width)
        layers_size.insert(-i, current_width)
        current_width = trial.suggest_int(f"width_{i}", bottleneck, current_width - 1) # -1 to exclude upper bound
        i += 1

    layers_size.insert(0, input_size)
    layers_size.append(output_size)

    return layers_size

 

def objective(trial, data, model_name, model_configs):
    # Sample the model parameters
    define_model = eval(f"define_{model_name}")
    suggest_dict = {
        "weight_decay": trial.suggest_float("weight_decay", cnst.DECAY_RANGE[0], cnst.DECAY_RANGE[1], log=True),
        "lr": trial.suggest_float("lr", cnst.LR_RANGE[0], cnst.LR_RANGE[1]),
        "layers_size": define_model(trial, data)
    }
    

    # We'll use Strategy Acquisition class JUST to train the tuned model even if it's not an upstream model
    idx_lb = np.arange(len(data["train"])) # all indices of train since we use it for the upstream training and not AL
    model_configs.update(suggest_dict)
    base_strategy = Strategy(upstream_arch=MLP, upstream_configs=model_configs, data=data, idx_lb=idx_lb)
    base_strategy.train_upstream(trial=trial)
    loss, metrics = base_strategy.eval("val")

    return loss

def params_wrapper(best_params, model_configs, model="MLP"):
    decoded_params = {
        "lr": best_params["lr"],
        "weight_decay": best_params["weight_decay"]
    }
  
    width_dict = {k: v for k, v in best_params.items() if "width_" in k}
    layers_size = []
    for idx, _ in enumerate(width_dict.keys()):
        layers_size.append(width_dict[f"width_{idx}"])
    if model == "AE":
        # TODO check whether it works correctly
        layers_size.extend(layers_size[1::-1])
    decoded_params["layers_size"] = layers_size
    decoded_params["layers_size"].append(model_configs["layers_size"][-1])
    decoded_params["layers_size"].insert(0, model_configs["layers_size"][0])

    return decoded_params

def hypers_search(data, model_arch_name, model_configs):
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_pruner = getattr(optuna.pruners, cnst.OPTUNA_PRUNER["name"])(**cnst.OPTUNA_PRUNER["configs"])
    optuna_sampler = getattr(optuna.samplers, cnst.OPTUNA_SAMPLER["name"])(seed=cnst.RANDOM_STATE, 
                                                                           **cnst.OPTUNA_SAMPLER["configs"])
    #TODO check the objective direction
    study = optuna.create_study(direction="minimize", sampler=optuna_sampler, pruner=optuna_pruner)
    wrapper_func = lambda trial: objective(trial, data, model_arch_name, model_configs=model_configs)
    study.optimize(wrapper_func, n_trials=cnst.TOTAL_TRIALS)

    return params_wrapper(study.best_params, model_configs)