import os

import torch
import numpy as np

from utilities import NN, ReplayBuffer, makedir
from datasets import VectoralDataset, ReplayDataset
from acquisitions import Acquisition
from core import Learnable, Pool

class Keychain(Acquisition):

    meta_arch = NN

    def __init__(self, buffer_capacity=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.meta_val_perf = []



    def get_scores(self, values=None):
        if values is None:
            self.keychain_iteration()
            self.meta_acq.tune_model()
            self.meta_val_perf.append(self.meta_acq.eval_model("val")[0])
            values, y = self.pool.get("unlabeled")

        inputs = self.collect_inputs(values)
        scores = self.meta_acq(torch.Tensor(inputs))
        return scores[:, 0] 
    
          
    def collect_inputs(self, x):
        probs = self.clf(torch.Tensor(x))
        values = np.concatenate((x, probs.cpu()), axis=-1)
        return values

    def keychain_iteration(self):
        best_loss, best_metrics = self.clf.eval_model("val")

        model_path = os.getcwd() + "/temp/"
        makedir(model_path)
        self.raw_targets = np.zeros((self.pool.get_len("labeled"), 1))

        model_path += f"keychain_model_{self.random_seed}_{self.pool.data['train'].__class__.__name__}"
        
        torch.save(self.clf.model.state_dict(), model_path)

        intact_labeled_pool = self.pool.idx_lb.copy()
        for idx in range(len(intact_labeled_pool)):
            self.pool.idx_lb = np.delete(intact_labeled_pool, idx)
            self.clf.reset_model()
            self.clf.train_model()

            loss, metrics = self.clf.eval_model("val")
            self.raw_targets[idx] += loss - best_loss

            self.pool.idx_lb = intact_labeled_pool.copy()
        

        self.clf.model.load_state_dict(torch.load(model_path))
        os.remove(model_path)
        
        x, y = self.pool.get("labeled")
        inputs = self.collect_inputs(x)
        targets = self.raw_targets.astype(np.float32)

        self.buffer.push(inputs, targets)

        self.soak_from_buffer()

    def soak_from_buffer(self):
        x, y = self.buffer.get_data()
        val_chunk = len(self.raw_targets)
        if len(self.buffer._y) == 1:
            train_idx, val_idx = VectoralDataset.conv_split(y.shape[0], shares=[0.6])
        else:
            train_idx, val_idx = VectoralDataset.step_split(y.shape[0], val_chunk)

        data = {
            "train": ReplayDataset(x[train_idx], y[train_idx]),
            "val": ReplayDataset(x[val_idx], y[val_idx])
        }
        pool = Pool(data=data, random_seed=self.random_seed)
        self.meta_acq = Learnable(pool=pool, random_seed=self.random_seed) 
