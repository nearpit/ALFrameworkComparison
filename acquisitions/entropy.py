from acquisitions import Acquisition
import torch
import numpy as np

class Entropy(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self, values=None):
        if values is None:
            values, y = self.pool.get("unlabeled")
        probs = self.clf(torch.Tensor(values)).cpu()
        log_probs = torch.log(probs + np.finfo(np.float32).smallest_normal)
        U = -(probs*log_probs).sum(axis=1)
        return U
