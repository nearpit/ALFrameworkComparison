from acquisitions import Acquisition
import torch

class Bald(Acquisition):
    def __init__(self, forward_passes=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
    
    def get_scores(self):
        self.clf.model.train()        # To enable Dropout 
        total_predictions = torch.empty((0, self.pool.get_len("unlabeled"), self.pool.n_classes)).to(self.clf.device)
        x, y = self.pool.get("unlabeled")
        for _ in range(self.forward_passes):
            probs = self.clf(torch.Tensor(x))
            total_predictions = torch.cat((total_predictions, torch.unsqueeze(probs, 0)))

        self.clf.model.eval()        # To disable Dropout 
        
        average_prob =  total_predictions.mean(dim=0)
        total_uncertainty = -(average_prob*torch.log(average_prob)).sum(dim=-1)
        data_uncertainty = (-(total_predictions*torch.log(total_predictions))).sum(dim=-1).mean(dim=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty
        return knowledge_uncertainty
