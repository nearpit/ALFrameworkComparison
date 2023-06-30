import torch

from acquisitions import Strategy
from utilities.backbones import EarlyStopper
import utilities.constants as cnst

# Active Learning via Reconstruction Model
class Alrm(Strategy):
    def __init__(self, rm_arch, rm_configs, meta_arch, meta_configs, model_path="models/meta_acq", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rm = rm_arch(**rm_configs) # rm - reconstruction model
        self.meta_acq = meta_arch(**meta_configs)
        self.model.load_state_dict(torch.load(model_path))
    
    @Strategy.reset_model(model_name="rm")
    def get_scores(self):
        self.train_rm()
        with torch.no_grad():
            inputs = self.get_unlabeled()[0].to(self.device)
            predictions = self.model(inputs)
            _, ulb_loss = self.eval_rm("ulb")

        meta_input = torch.cat((predictions, ulb_loss))
        scores = self.meta_acq(meta_input)
        return scores

    def eval_rm(self, split_name):
        total_loss = 0
        metric = self.rm.metric(device=self.device)
        loader = getattr(self, f"{split_name}_loader")

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = torch.cat((inputs, labels), dim=-1)
                labels = inputs.clone()

                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                predictions = self.model(inputs)

                batch_loss = self.model.criterion(predictions, labels)
                total_loss += batch_loss.item()
                metric.update(input=predictions.ravel(), target=labels.ravel())
        return total_loss, metric.compute().item()

    def train_rm(self):
        self.rm.eval()
        
        early_stopper = EarlyStopper()

        for _ in range(self.epochs):
            total_loss_train = 0

            train_metric =  self.rm.metric(device=self.device)
            val_metric =  self.rm.metric(device=self.device)

            for inputs, labels in self.train_loader:

                inputs = torch.cat((inputs, labels), dim=-1)
                labels = inputs.clone()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                predictions = self.rm(inputs.float())
                
                batch_loss = self.rm.criterion(predictions, labels)
                total_loss_train += batch_loss.item()

                train_metric.update(input=predictions.ravel(), target=labels.ravel())
                self.rm.zero_grad()
                batch_loss.backward()
                self.rm.optimizer.step()
            
                       
            total_acc_train = train_metric.compute()
            loss_val, acc_val = self.eval_rm("val")

            if early_stopper.early_stop(loss_val):
                break