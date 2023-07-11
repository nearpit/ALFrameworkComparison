import torch
import numpy as np

class Acquisition: 
    latent = None

    def __init__(self, 
                 clf,
                 pool,
                 random_seed):
        self.clf = clf       
        self.pool = pool
        self.random_seed = random_seed

    def get_scores(self):
        pass

    def query(self):
        all_scores = self.get_scores()
        max_scores = np.argwhere(all_scores == all_scores.max()).ravel()
        idx = np.random.choice(max_scores, 1)[0]
        return self.pool.idx_ulb[idx], idx, all_scores 
    
    # auxiliary function for latent representations
    def get_activation(self, name):
        def hook(model, input, output):
            value = torch.clone(output.detach())
            self.latent = value
        return hook

    def embedding_hook(self): # define the hooked layers if needed
        pass
    