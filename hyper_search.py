import json
import optuna
import torch

from optuna.trial import TrialState

from utilities import funcs, constants as cnst
from utilities.models import MLP
from utilities.classes import EarlyStopper

from datasets_manipulations import load_clean_dataset, generate_toy

directions = {
        "MLP": "maximize",
        "AE": "minimize"
}

def define_MLP(trial):
    input_size, output_size = train.dimensions
    weight_decay = trial.suggest_float("weight_decay", cnst.DECAY_RANGE[0], cnst.DECAY_RANGE[1], log=True)
    depth = trial.suggest_int("depth", cnst.DEPTH_RANGE[0], cnst.DEPTH_RANGE[1])
    lr = trial.suggest_float("lr", cnst.LR_RANGE[0], cnst.LR_RANGE[1])
    layers_size = [input_size]
    for idx in range(depth):
        layers_size.append(trial.suggest_int(f"width_{idx}", cnst.WIDHT_RANGE[0], cnst.WIDHT_RANGE[1]))
    layers_size.append(output_size)
    model = MLP(layers_size=layers_size, last_layer=configs["final_activation"]).to(DEVICE)
    optimizer = getattr(torch.optim, cnst.OPTIMIZER)(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer

def define_AE(trial):
    input_size, output_size = train.dimensions
    weight_decay = trial.suggest_float("weight_decay", cnst.DECAY_RANGE[0], cnst.DECAY_RANGE[1], log=True)
    max_width = trial.suggest_int("max_width", cnst.WIDHT_RANGE[0], cnst.WIDHT_RANGE[1])
    bottleneck =  trial.suggest_int("bottleneck", cnst.WIDHT_RANGE[0] - 1, max_width - 1)
    lr = trial.suggest_float("lr", cnst.LR_RANGE[0], cnst.LR_RANGE[1])
    layers_size = [max_width, bottleneck, max_width]
    current_width = trial.suggest_int(f"width_{0}", bottleneck, max_width - 1) # -1 to exclude upper bound

    i = 1

    while current_width > bottleneck:
        layers_size.insert(i, current_width)
        layers_size.insert(-i, current_width)
        current_width = trial.suggest_int(f"width_{i}", bottleneck, current_width - 1) # -1 to exclude upper bound
        i += 1
       
    layers_size.insert(0, input_size)
    layers_size.append(output_size)

    model = MLP(layers_size=layers_size, last_layer="Identity").to(DEVICE)
    optimizer = getattr(torch.optim, cnst.OPTIMIZER)(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer   


def objective(trial):

    early_stopper = EarlyStopper(patience=cnst.PATIENCE, min_delta=cnst.MIN_DELTA)
    define_model = eval(f"define_{args.tuned_model}")
    model, optimizer = define_model(trial)


    if args.tuned_model == "MLP":
        criterion = getattr(torch.nn, configs["loss"])()
        metric = configs["metric"]
    elif args.tuned_model == "AE":
        criterion = torch.nn.MSELoss()
        metric = "MeanSquaredError"



    for epoch_num in range(cnst.EPOCHS):
        loss, metric_val, model = funcs.train(model, train_loader, val_loader, criterion, optimizer, metric, DEVICE)
        trial.report(metric_val, epoch_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if early_stopper.early_stop(loss):             
            break
    return metric_val
    



if __name__ == '__main__':
    generate_toy()
    args = funcs.get_arguments()
    configs = funcs.get_configs(args.dataset)

    data = load_clean_dataset(dataset_name=args.dataset)
    train, val, test = data['train'], data['val'], data['test']
    train_loader = torch.utils.data.DataLoader(train, batch_size=configs['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=configs['batch_size'])
    test_loader = torch.utils.data.DataLoader(test, batch_size=configs['batch_size'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=cnst.N_STARTUP_TRIALS, n_warmup_steps= cnst.N_WARMUP_STEPS)

    study = optuna.create_study(direction=directions[args.tuned_model], sampler=optuna.samplers.TPESampler(), pruner=optuna_pruner)
    study.optimize(objective, n_trials=cnst.TOTAL_TRIALS)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    # Serializing json
    json_object = json.dumps(trial.params, indent=4)
    
    # Writing to sample.json
    with open(f"{args.dataset}_{args.tuned_model}_hypers.json", "w") as outfile:
        outfile.write(json_object)
