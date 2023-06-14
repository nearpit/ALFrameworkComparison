import json
import optuna
import torch

from optuna.trial import TrialState

from utilities import funcs, constants as cnst
from utilities.models import MLP 
from utilities.classes import EarlyStopper

from datasets_manipulations import load_clean_dataset, generate_toy

def define_model(trial):
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

def objective(trial):
    model, optimizer = define_model(trial)
    criterion = getattr(torch.nn, configs["loss"])()
    metric = configs["metric"]
    early_stopper = EarlyStopper(patience=cnst.PATIENCE, min_delta=cnst.MIN_DELTA)

    for epoch_num in range(cnst.EPOCHS):
        loss, accuracy = funcs.train_evaluate(model, train_loader, val_loader, criterion, optimizer, metric, DEVICE)
        trial.report(accuracy, epoch_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if early_stopper.early_stop(loss):             
            break
    return accuracy
    



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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna_pruner)
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
    with open(f"{args.dataset}_hypers.json", "w") as outfile:
        outfile.write(json_object)
