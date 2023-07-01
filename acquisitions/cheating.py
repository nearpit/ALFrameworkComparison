import os

import torch
import numpy as np

from acquisitions import Strategy
from utilities import constants as cnst

class Cheating(Strategy):
                       #DEBUG 
    def __init__(self, sample_size=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
    
    def get_scores(self):
        batch = np.random.choice(self.idx_ulb, self.sample_size, replace=False)
        scores = np.full((len(self.idx_intact)), float("-inf"))
        model_path = os.getcwd() + "/temp/"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path +=  "/oracle_model"
        torch.save(self.model.state_dict(), model_path)

        for candidate in batch:
            self.reset_model()
            self.add_new_inst(candidate)
            self.train_model()
            loss, accuracy = self.eval_model('val')
            scores[candidate] = -loss # to revert the loss values in order to align argmax query
            self.idx_lb = self.idx_lb[:-1] # removing just added candidate
        
        return scores[self.idx_ulb]