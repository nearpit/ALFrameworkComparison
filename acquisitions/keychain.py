import os

import torch
import numpy as np

from utilities import NN, ReplayBuffer
from datasets import VectoralDataset, ReplayDataset
from acquisitions import Acquisition
from core import Learnable, Pool

class Keychain(Acquisition):

    meta_arch = NN

    def __init__(self, buffer_capacity=5, forward_passes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
        self.buffer = ReplayBuffer(buffer_capacity)

    def get_scores(self):
        self.keychain_iteration()
        self.meta_acq.train_model()
        x, y = self.pool.get("unlabeled")
        scores = self.meta_acq(torch.Tensor(x))
        return scores[:, 0] 
                
    
    def keychain_iteration(self):
        best_loss, best_metrics = self.clf.eval_model("val")

        model_path = os.getcwd() + "/temp/keychain_model"
        torch.save(self.clf.model.state_dict(), model_path)
        inputs, targets = [], []
        labeled_pool = self.pool.idx_lb.copy()

        for idx, instance in enumerate(labeled_pool):
            self.idx_lb = np.delete(labeled_pool, idx)
            x, y = self.pool.data["train"][instance]
            self.clf.reset_model()
            self.clf.train_model()
            loss, metrics = self.clf.eval_model("val")
            # with torch.no_grad(): # PROBS
            #     probs = self.model(torch.Tensor(x))
            inputs.append(torch.Tensor(x))
            targets.append(torch.Tensor([max(0, loss - best_loss)]))

            self.pool.idx_lb = labeled_pool.copy()


        self.clf.model.load_state_dict(torch.load(model_path))

        self.buffer.push((inputs, targets))

        self.soak_from_buffer()


    def soak_from_buffer(self):
        x, y = self.buffer.get_data()
        train_idx, val_idx = VectoralDataset.conv_split(y.shape[0], shares=[0.8])
        data = {
            "train": ReplayDataset(x[train_idx], y[train_idx]),
            "val": ReplayDataset(x[val_idx], y[val_idx])
        }
        pool = Pool(data=data, random_seed=self.random_seed)
        self.meta_acq = Learnable(pool=pool, random_seed=self.random_seed) 