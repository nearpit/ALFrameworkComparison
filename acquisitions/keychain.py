import copy
import torch
import numpy as np
from collections import Counter

from core import Learnable, Pool
from utilities import ReplayBuffer, NN
from datasets import ReplayDataset, AutoEncoderDataset
from acquisitions import Acquisition

class KeychainBase(Acquisition):

    meta_arch = NN
    n_meta_trials = 50 #DEBUG
                                            #DEBUG
    def __init__(self, buffer_capacity=1, forward_passes=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.meta_val_perf = []

    @property
    def total_repititions(self):
        return self.budget - self.pool.get_len("new_labeled")
    
    def collect_inputs(self):
        pass


    def keychain_iteration(self):
        current_pool = copy.copy(self.pool)
        intact_new_labeled_idx = copy.copy(current_pool.idx_new_lb)
        playground_clf = copy.deepcopy(self.clf)
        
        unviolated_train_idx, unviolated_val_idx = next(current_pool.get_unviolated_splitter(tune=False))
        _, val_loader = current_pool.get_train_val_loaders(unviolated_train_idx, unviolated_val_idx)
        genuine_val_loss, _ = self.clf.eval(val_loader)

        abs_idx_unviolated_train = self.pool.idx_unviolated_lb[unviolated_train_idx]
        idx_train = np.sort(np.append(intact_new_labeled_idx, abs_idx_unviolated_train))
        scores = np.zeros((len(idx_train)))
        probs = np.full(len(idx_train), 1/len(idx_train))
        for i in range(self.forward_passes):
            np.random.seed(i)
            augmented_train_idx = np.random.choice(idx_train, self.total_repititions, p=probs)
            total_train_idx = np.concatenate((idx_train, augmented_train_idx))
            counter_list = np.array([x[1] for x in sorted(Counter(total_train_idx).items())], dtype=np.float32) # count the amount of repetitions of each instance in the iteration
            current_pool.idx_new_lb = total_train_idx
            playground_clf.pool = current_pool
            train_loader, val_loader = current_pool.get_train_val_loaders([], unviolated_val_idx)
            _, val_perf = playground_clf.fit(train_loader=train_loader, val_loader=val_loader)
            gain = 1 - val_perf[0]/genuine_val_loss
            scores += counter_list*float(gain)
            min_score = np.min(scores)
            shifted_scores = scores - min_score + 1e-9 # small epsilon to make sure that all values are positives
            probs = shifted_scores/shifted_scores.sum()

        x, _ = self.pool[idx_train]
        inputs = self.collect_inputs(x)
        targets = scores.astype(np.float32).copy()
        self.buffer.push(inputs, targets)
        print(scores)
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
        pool = Pool(data=data, args=self.pool.args, val_share=0.3, n_initially_labeled=-1)
        self.meta_acq = Learnable(pool=pool, random_seed=self.random_seed, model_arch_name="MLP_reg") 

class Keychain_naive(KeychainBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_scores(self, values=None):
        if values is None:
            self.keychain_iteration()
            train_perf, val_perf, _ = self.meta_acq.tune_model(n_trials=self.n_meta_trials, online=True)
            self.meta_val_perf.append(val_perf[0])
            values, y = self.pool.get("unlabeled")

        inputs = self.collect_inputs(values)
        scores = self.meta_acq(torch.Tensor(inputs)).cpu()
        return scores[:, 0] 
    
          
    def collect_inputs(self, x):
        probs = self.clf(torch.Tensor(x))
        values = np.concatenate((x, probs.cpu()), axis=-1)
        return values

class Keychain_autoencoder(KeychainBase):
    ae_n_trials = 25 #DEBUG

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ae_val_perf = []

    def get_scores(self, values=None):
        if values is None:
            self.update_ae()
            self.keychain_iteration()
            train_perf, val_perf, _ = self.meta_acq.tune_model(n_trials=self.n_meta_trials, online=True)
            self.meta_val_perf.append(val_perf[0])
            values, y = self.pool.get("unlabeled")

        inputs = self.collect_inputs(values)
        scores = self.meta_acq(torch.Tensor(inputs)).cpu()
        return scores[:, 0] 
    

    def update_ae(self):
        inputs, _ = self.pool.get("all_labeled")
        targets = self.clf(torch.Tensor(inputs)).cpu()
        x = np.column_stack((inputs, targets))
        data = {"train": AutoEncoderDataset(x, self.pool.batch_size)}
        pool = Pool(data=data, args=self.pool.args, n_initially_labeled=-1)
        self.ae = Learnable(pool=pool, random_seed=self.random_seed, model_arch_name="AE") 
        train_perf, val_perf, _ = self.ae.tune_model(n_trials=self.ae_n_trials, online=True)
        self.ae_val_perf.append(val_perf[0])

    def collect_inputs(self, x):
        probs = self.clf(torch.Tensor(x))
        values = torch.cat((torch.Tensor(x).to(self.clf.device), probs), dim=-1)
        ae_outputs = self.ae(torch.Tensor(values))
        ae_loss = torch.pow((ae_outputs - values).sum(dim=-1), 2).reshape(-1, 1)
        collected_values = np.concatenate((x, probs.cpu(), ae_loss.cpu()), axis=-1)
        return collected_values
