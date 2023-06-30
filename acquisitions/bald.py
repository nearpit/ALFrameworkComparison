from acquisitions import Strategy
import torch

class Bald(Strategy):
    def __init__(self, forward_passes=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_passes = forward_passes
    
    def get_scores(self):
        self.model.train()        # To enable Dropout 
        n_classes = self.model_configs["layers_size"][-1]
        total_predictions = torch.empty((0, len(self.idx_ulb), n_classes))
        for _ in range(self.forward_passes):
            with torch.no_grad():
                probs = self.model(torch.Tensor(self.train_dataset[self.idx_ulb][0]))
            total_predictions = torch.cat((total_predictions, torch.unsqueeze(probs, 0)))

        self.model.eval()        # To disable Dropout 
        
        average_prob =  total_predictions.mean(dim=0)
        total_uncertainty = -(average_prob*torch.log(average_prob)).sum(dim=-1)
        data_uncertainty = (-(total_predictions*torch.log(total_predictions))).sum(dim=-1).mean(dim=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty
        return knowledge_uncertainty
    