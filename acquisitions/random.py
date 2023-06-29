from acquisitions import Strategy
import numpy as np

class Random(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self):
        return np.random.random(self.idx_ulb)