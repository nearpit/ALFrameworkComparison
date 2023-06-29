import os

import torch
import numpy as np

from acquisitions import Strategy
from utilities import constants as cnst

class Cheating(Strategy):
    def __init__(self, sample_size=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
    
    def query(self):
        batch = np.random.choice(self.idx_ulb, self.sample_size, replace=False)
        best_score = float("Inf")

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
            if loss <= best_score:
                best_score = loss
                best_candidate = candidate
            self.idx_lb = self.idx_lb[:-1] # removing just added candidate
        
        return best_candidate