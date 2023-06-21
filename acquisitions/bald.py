from acquisitions import Strategy
import torch

class Bald(Strategy):
    def __init__(self, forward_passes=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
    
    def query(self):
        self.clf.train()        # To enable Dropout 
        n_classes = self.model_configs["layers_size"][-1]
        if n_classes == 1:
            n_classes += 1
        total_predictions = torch.empty((0, len(self.idx_ulb), n_classes))
        for _ in range(self.forward_passes):
            with torch.no_grad():
                probs = self.clf(torch.Tensor(self.train_dataset[self.idx_ulb][0]))
                if probs.shape[1] == 1:
                    probs = torch.cat((probs, 1-probs), axis=-1)
            total_predictions = torch.cat((total_predictions, torch.unsqueeze(probs, 0)))
        average_prob =  total_predictions.mean(dim=0)
        total_uncertainty = -(average_prob*torch.log(average_prob)).sum(dim=-1)
        data_uncertainty = (-(total_predictions*torch.log(total_predictions))).sum(dim=-1).mean(dim=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty
        return self.idx_ulb[knowledge_uncertainty.argsort()[-1]]