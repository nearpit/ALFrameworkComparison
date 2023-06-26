from acquisitions import Strategy
import torch

class Entropy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def query(self):
        with torch.no_grad():
            probs = self.upstream_model(torch.Tensor(self.train_dataset[self.idx_ulb][0]))
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(axis=1)
        sorted_entropy = U.argsort()
        return self.idx_ulb[sorted_entropy[-1]]