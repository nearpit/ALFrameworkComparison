from acquisitions import Acquisition
import torch


class NormalizingFlow(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_scores(self, values=None):
        if values is None:
            values, _ = self.pool.get("unlabeled")
        probs = self.clf(torch.Tensor(values)).cpu()
        log_probs = torch.log(probs + torch.finfo(torch.float32).smallest_normal)
        U = -(probs * log_probs).sum(axis=1)
        return U

    def prepare_U_flow(self):
        pass

    def train_L_flow(self):
        pass
