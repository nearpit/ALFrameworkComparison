from acquisitions import Acquisition
import numpy as np

class Random(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self, values=None):
        if values is None:
            values = self.pool.get_len("unlabeled")
        else:
            values = values[:, 0].ravel().shape[0]
        return np.random.random(values)