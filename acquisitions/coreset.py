import torch 
import numpy as np
from sklearn.metrics import pairwise_distances

from acquisitions import Acquisition
from core import Learnable

class Coreset(Acquisition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # @Learnable.hook_once    
    def get_scores(self, values=None):
        if values is None:
            values = self.pool.get("unlabeled")[0]

        latent_ulb = self.get_embeddings(values).clone()
        latent_lb = self.get_embeddings(self.pool.get("all_labeled")[0]).clone()
        pair_distance = pairwise_distances(latent_ulb.cpu(), latent_lb.cpu())
        min_dist = np.amin(pair_distance, axis=1)

        return min_dist

    def get_embeddings(self, inputs):
        self.clf(torch.Tensor(inputs))
        return self.clf.latent