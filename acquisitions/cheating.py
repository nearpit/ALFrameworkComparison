import os

import torch
import numpy as np

from acquisitions import Acquisition

class Cheating(Acquisition):
                       #DEBUG 
    def __init__(self, sample_size=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
    
    def get_scores(self):
        batch = np.random.choice(self.pool.idx_ulb, self.sample_size, replace=False)
        scores = np.full(self.pool.get_len(), float("-inf"))
        model_path = os.getcwd() + "/temp/"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path +=  "/oracle_model"
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

        return scores[self.pool.idx_ulb]
