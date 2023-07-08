from acquisitions import Acquisition
import torch

class Entropy(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self):
        x, y = self.pool.get("unlabeled")
        with torch.no_grad():
            probs = self.clf(torch.Tensor(x))
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(axis=1)
        return U
