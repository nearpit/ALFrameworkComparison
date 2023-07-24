from acquisitions import Acquisition
import torch

class Margin(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self, values=None):
        if values is None:
            values, y = self.pool.get("unlabeled")
        probs = self.clf(torch.Tensor(values)).cpu()
        sorted_probs, idx = probs.sort(descending=True)
        U = sorted_probs[:, 0] - sorted_probs[:, 1]
        return -U # in order to aligh to the argmax query
