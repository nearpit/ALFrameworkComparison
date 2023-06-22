import torch 
import numpy as np
from sklearn.metrics import pairwise_distances

from acquisitions import Strategy

class Coreset(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent = None
    
    @Strategy.hook_penultimate_once    
    def query(self):
        m = self.get_unlabeled()[0].shape[0]

        latent_ulb, latent_lb = self.get_embeddings(self.idx_ulb), self.get_embeddings(self.idx_lb)

        pair_distance = pairwise_distances(latent_ulb, latent_lb)
        min_dist = np.amin(pair_distance, axis=1)

        idx = min_dist.argmax()

        return self.idx_ulb[idx]

    def get_embeddings(self, set_indices):
        with torch.no_grad():
            self.clf(torch.Tensor(self.train_dataset[set_indices][0]))
        return self.latent
    
    def embedding_hook(self):
        total_layer_depth = len(self.clf_configs["layers_size"])
        penultimate_layer_name = f"dense_{total_layer_depth - 2}" 
        penultimate_layer = getattr(self.clf.layers, penultimate_layer_name)
        penultimate_layer.register_forward_hook(self.get_activation(penultimate_layer_name))
        
    def get_activation(self, name):
        def hook(model, input, output):
            value = torch.clone(output.detach())
            self.latent = value
        return hook
    
    