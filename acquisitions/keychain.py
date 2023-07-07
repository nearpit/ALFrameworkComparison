import os

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from utilities import NN, ReplayBuffer, OnlineAvg
from datasets import VectoralDataset, ReplayDataset
from acquisitions import Acquisition
from core import Learnable, Pool

class Keychain(Acquisition):

    meta_arch = NN

    def __init__(self, buffer_capacity=5, n_samples=40, batch_share=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.batch_share = batch_share
        self.buffer = ReplayBuffer(buffer_capacity)
        self.feature_encoder = FunctionTransformer
        self.target_encoder = FunctionTransformer

    def get_scores(self):
        self.keychain_iteration()
        self.meta_acq.train_model()
        x, y = self.pool.get("unlabeled")
        inputs = self.preprocess(self.collect_inputs(x), self.feature_encoder())
        with torch.no_grad():
            scores = self.meta_acq(inputs)
        return scores[:, 0] 
    
    def preprocess(self, x, encoder):
        x_transformed = encoder.fit_transform(x)
        return torch.from_numpy(x_transformed)
           
    def collect_inputs(self, x):
        x = torch.Tensor(x)
        with torch.no_grad():
            probs = self.clf.model(x)
        values = np.concatenate((x, probs), axis=-1)
        return values

    def keychain_iteration(self):
        best_loss, best_metrics = self.clf.eval_model("val")

        model_path = os.getcwd() + "/temp/keychain_model"
        torch.save(self.clf.model.state_dict(), model_path)

        raw_targets = np.vectorize(lambda x: OnlineAvg())(np.zeros((self.pool.get_len("labeled"), 1)))
        labeled_pool = self.pool.idx_lb.copy()
        relative_idx = np.arange(self.pool.get_len("labeled"))
        for _ in range(self.n_samples):
            temp_removed = np.random.choice(relative_idx, int(len(labeled_pool)*self.batch_share), replace=False)
            self.pool.idx_lb = np.delete(labeled_pool, temp_removed)
            self.clf.reset_model()
            self.clf.train_model()

            loss, metrics = self.clf.eval_model("val")
            raw_targets[temp_removed] += max(0, loss - best_loss)

            self.pool.idx_lb = labeled_pool.copy()
        

        self.clf.model.load_state_dict(torch.load(model_path))
        
        x, y = self.pool.get("labeled")
        inputs = self.preprocess(self.collect_inputs(x), self.feature_encoder(lambda x: x))
        targets = self.preprocess(raw_targets.astype(np.float32), self.target_encoder(lambda x: x))

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
        self.meta_acq = Learnable(pool=pool, random_seed=self.random_seed, model_arch_name="MLP_reg") 