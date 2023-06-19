from acquisitions import Strategy
from utilities import constants as cnst
import numpy as np

class RandomSampler(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def query(self):
        return np.random.choice(self.idx_ulb, 1)