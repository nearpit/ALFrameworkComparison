from acquisitions import Acquisition
import torch

class Margin(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self):
        x, y = self.pool.get("unlabeled")
        with torch.no_grad():
            probs = self.clf(torch.Tensor(x))
        sorted_probs, idx = probs.sort(descending=True)
        U = sorted_probs[:, 0] - sorted_probs[:, 1]
        return -U # in order to aligh to the argmax query