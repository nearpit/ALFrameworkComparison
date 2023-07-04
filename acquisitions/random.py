from acquisitions import Acquisition
import numpy as np

class Random(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self):
        return np.random.random(self.pool.get_len("unlabeled"))