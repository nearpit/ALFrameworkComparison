from acquisitions import Strategy
import numpy as np

class Random(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def query(self):
        return np.random.choice(self.idx_ulb, 1)[0]