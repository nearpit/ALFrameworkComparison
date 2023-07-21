import copy
import torch
import numpy as np

from core import Learnable, Pool
from utilities import ReplayBuffer, NN, OnlineAvg
from datasets import ReplayDataset
from acquisitions import Acquisition

class Keychain(Acquisition):

    meta_arch = NN
    n_meta_trials = 25 # DEBUG
                                            # DEBUG
    def __init__(self, buffer_capacity=1, forward_passes=25, sample_share=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.forward_passes = forward_passes
        self.sample_share = sample_share
        self.meta_val_perf = []


    def get_scores(self, values=None):
        if values is None:
            self.keychain_iteration()
            train_perf, val_perf, _ = self.meta_acq.tune_model(n_trials=self.n_meta_trials, online=True)
            self.meta_val_perf.append(val_perf[0])
            values, y = self.pool.get("unlabeled")

        inputs = self.collect_inputs(values)
        scores = self.meta_acq(torch.Tensor(inputs))
        return scores[:, 0] 
    
          
    def collect_inputs(self, x):
        probs = self.clf(torch.Tensor(x))
        values = np.concatenate((x, probs.cpu()), axis=-1)
        return values

    def keychain_iteration(self):
        
        current_pool = copy.copy(self.pool)
        intact_new_labeled_idx = copy.copy(current_pool.idx_new_lb)
        playground_clf = copy.deepcopy(self.clf)
        unviolated_train_idx, unviolated_val_idx = next(current_pool.get_unviolated_splitter(tune=False))

        _, val_loader = current_pool.get_train_val_loaders(unviolated_train_idx, unviolated_val_idx)
        best_loss, _ = self.clf.eval(val_loader)

        relative_unviolated_idx = np.arange(unviolated_train_idx.shape[0])
        relative_new_idx = np.arange(intact_new_labeled_idx.shape[0])

        self.raw_targets_unviolated = np.vectorize(lambda x: OnlineAvg(1))(np.zeros((len(unviolated_train_idx), 1)))
        if len(relative_new_idx):
            self.raw_targets_new = np.vectorize(lambda x: OnlineAvg(1))(np.zeros((len(intact_new_labeled_idx), 1)))
        else:
            self.raw_targets_new = np.empty((0, 1))

        for idx in range(self.forward_passes):
            np.random.seed(idx)
            
            if len(relative_new_idx):
                extra_new_idx  = np.random.choice(relative_new_idx, int(len(relative_new_idx)*self.sample_share), replace=False)
                sample_new_idx = np.append(relative_new_idx, extra_new_idx)
            else:
                extra_new_idx = np.array([], dtype=int)
                sample_new_idx = np.array([], dtype=int)

            current_pool.idx_new_lb = intact_new_labeled_idx[sample_new_idx]
            playground_clf.pool = current_pool

            extra_unviolated_idx = np.random.choice(relative_unviolated_idx, int(len(relative_unviolated_idx)*self.sample_share), replace=False)
            sample_unviolated_idx = np.append(relative_unviolated_idx, extra_unviolated_idx)
            temp_unviolated_train_idx = unviolated_train_idx[sample_unviolated_idx]

            train_loader, val_loader = current_pool.get_train_val_loaders(temp_unviolated_train_idx, unviolated_val_idx)
            train_perf, val_perf = playground_clf.fit(train_loader=train_loader, val_loader=val_loader)
            loss = best_loss/val_perf[0]

            self.raw_targets_unviolated[extra_unviolated_idx] += loss
            self.raw_targets_new[extra_new_idx] += loss
        

        x, _ = self.pool.get("unviolated")
        x_unv = x[unviolated_train_idx]

        x, _ = self.pool.get("new_labeled")
        x = np.concatenate((x_unv, x))
        inputs = self.collect_inputs(x)


        targets = np.concatenate((self.raw_targets_unviolated.astype(np.float32), self.raw_targets_new.astype(np.float32)))

        self.buffer.push(inputs, targets)

        self.soak_from_buffer()

    def get_probs(self, array):
        new_idx_prob = array.astype(int)
        new_idx_prob = new_idx_prob/new_idx_prob.sum()
        return new_idx_prob.ravel()

    def soak_from_buffer(self):
        x, y = self.buffer.get_data()
        data = {
            "train": ReplayDataset(x, y)
        }
        pool = Pool(data=data, args=self.pool.args, val_share=0.3, n_initially_labeled=y.shape[0])
        self.meta_acq = Learnable(pool=pool, random_seed=self.random_seed) 