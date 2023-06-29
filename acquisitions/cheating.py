import os

import torch
import numpy as np

from acquisitions import Strategy
from utilities import constants as cnst

class Cheating(Strategy):
    def __init__(self, sample_size=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
    
    def get_scores(self):
        batch = np.random.choice(self.idx_ulb, self.sample_size, replace=False)
        scores = np.full((len(self.idx_intact)), -1.)
        model_path = os.getcwd() + "/temp/"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path +=  "/oracle_model"
        torch.save(self.upstream_model.state_dict(), model_path)

        for candidate in batch:
            self.upstream_model = self.initialize_upstream()
            self.upstream_model.load_state_dict(torch.load(model_path))
            self.update(candidate)
            self.train_upstream()
            loss, accuracy = self.eval('val')
            scores[candidate] = 1 / (np.finfo(np.float32).eps + loss) # to revert the loss values in order to align argmax query
            self.idx_lb = self.idx_lb[:-1] # removing just added candidate
        
        return scores[self.idx_ulb]