import json
import optuna
import torch
import numpy as np 

from optuna.trial import TrialState
from torcheval import metrics

from utilities import funcs, constants as cnst
from utilities.models import MLP
from utilities.classes import EarlyStopper

from datasets_manipulations import load_clean_dataset

args = funcs.get_arguments()
configs = funcs.get_configs(args.dataset)

data = load_clean_dataset(dataset_name=args.dataset)
train, val, test = data['train'], data['val'], data['test']
train_loader = torch.utils.data.DataLoader(train, batch_size=configs['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=configs['batch_size'])
test_loader = torch.utils.data.DataLoader(test, batch_size=configs['batch_size'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

early_stopper = EarlyStopper(patience=cnst.PATIENCE, min_delta=cnst.MIN_DELTA)

input_size, output_size = train.dimensions
layers_size = [input_size]
layers_size.extend(configs["MLP_configs"]["layers_size"])
layers_size.append(output_size)

model = MLP(layers_size=layers_size, last_layer=configs["final_activation"]).to(DEVICE)
optimizer = getattr(torch.optim, cnst.OPTIMIZER)(model.parameters(), lr=configs["MLP_configs"]["lr"], weight_decay=configs["MLP_configs"]["weight_decay"])
criterion = getattr(torch.nn, configs["loss"])()
metric_name = configs["metric"]

for epoch_num in range(cnst.EPOCHS):
    loss, accuracy, model = funcs.train(model, train_loader, val_loader, criterion, optimizer, metric_name, DEVICE)
    if early_stopper.early_stop(loss):             
        with torch.no_grad():
            for split in ["train", "val", "test"]:
                loader = eval(f"{split}_loader")
                data = torch.empty(0, input_size + output_size)
                metric = getattr(metrics, metric_name)()
                for inputs, labels in loader:
                    labels = labels.to(DEVICE)
                    inputs = inputs.to(DEVICE)
                    output = model(inputs)
                    ae_inputs = torch.cat((inputs, output), dim=1)
                    data = torch.cat((data, ae_inputs))
                    batch_loss = criterion(output, labels)
                    metric.update(input=output.squeeze(), target=labels.squeeze())
                with open(f"datasets/{args.dataset}_AE/{split}", 'wb') as f:
                    data = data.numpy()
                    np.savez(f, x=data, y=data)
                print(split, metric.compute())
                
        break