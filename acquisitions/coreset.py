import torch 
import numpy as np
from sklearn.metrics import pairwise_distances

from acquisitions import Acquisition
from core import Learnable

class Coreset(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @Learnable.hook_once    
    def get_scores(self):
        latent_ulb, latent_lb = self.get_embeddings(self.pool.get("unlabeled")), self.get_embeddings(self.pool.get("labeled"))

        pair_distance = pairwise_distances(latent_ulb.cpu(), latent_lb.cpu())
        min_dist = np.amin(pair_distance, axis=1)

        return min_dist

    def get_embeddings(self, inputs):
        x, y = inputs
        self.clf(torch.Tensor(x))
        return self.latent
    
    def embedding_hook(self):
        # penultimate layer hook
        total_layer_depth = len(self.clf.model_configs["layers_size"])
        penultimate_layer_name = f"dense_{total_layer_depth - 2}" 
        penultimate_layer = getattr(self.clf.model.layers, penultimate_layer_name)
        penultimate_layer.register_forward_hook(self.get_activation(penultimate_layer_name))
        

    
    
