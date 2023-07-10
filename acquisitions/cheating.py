import os

import torch
import numpy as np

import utilities
from acquisitions import Acquisition

class Cheating(Acquisition):
                       #DEBUG 
    def __init__(self, sample_size=30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
    
    def get_scores(self):
        batch = np.random.choice(self.pool.idx_ulb, self.sample_size, replace=False)
        scores = np.full(self.pool.get_len(), float("-inf"))
        model_path = os.getcwd() + "/temp/"
        
        utilities.makedir(model_path)

        model_path +=  f"cheating_model_{self.random_seed}_{self.pool.data['train'].__class__.__name__}"
        torch.save(self.clf.model.state_dict(), model_path)

        for candidate in batch:
            torch.manual_seed(self.random_seed)
            self.clf.reset_model()
            self.pool.add_new_inst(candidate)
            self.clf.train_model()
            loss, accuracy = self.clf.eval_model('val')
            scores[candidate] = -loss # to revert the loss values in order to align argmax query
            self.pool.idx_lb = self.pool.idx_lb[:-1] # removing just added candidate
        
        self.clf.model.load_state_dict(torch.load(model_path)) # restoring the initial params
        os.remove(model_math)

        return scores[self.pool.idx_ulb]
