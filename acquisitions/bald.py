from acquisitions import Acquisition
import torch
import numpy as np

class Bald(Acquisition):
    def __init__(self, forward_passes=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
    
    def get_scores(self, values=None):

        if values is None:
           values = self.pool.get("unlabeled")[0]
        total_predictions = torch.empty((0, len(values), self.pool.n_classes)).to(self.clf.device)
        for _ in range(self.forward_passes):
            probs = self.clf(torch.Tensor(values), mc_dropout=True)
            total_predictions = torch.cat((total_predictions, torch.unsqueeze(probs, 0)))
        
        average_prob =  total_predictions.mean(dim=0)
        total_uncertainty = -(average_prob*torch.log(average_prob + np.finfo(np.float32).smallest_normal)).sum(dim=-1)
        data_uncertainty = (-(total_predictions*torch.log(total_predictions + np.finfo(np.float32).smallest_normal))).sum(dim=-1).mean(dim=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty
        return knowledge_uncertainty.cpu()
