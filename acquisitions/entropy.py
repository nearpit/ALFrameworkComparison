from acquisitions import Strategy
import torch

class Entropy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_scores(self):
        with torch.no_grad():
            probs = self.model(torch.Tensor(self.train_dataset[self.idx_ulb][0]))
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(axis=1)
        return U