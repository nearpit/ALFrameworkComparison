from sklearn.datasets import make_blobs

from datasets.toy import Toy

class Blobs(Toy):
    dataset_name = "blobs"
    std = 0.25
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_noise(self):
        return make_blobs(n_samples=self.configs["n_honeypot"], 
                          centers=[(1, 1)],
                          cluster_std=[1.5*self.std],
                          random_state=self.random_seed)

    
    def generate_clean(self):
        return make_blobs(n_samples=self.configs["n_instances"] - self.configs["n_honeypot"],
                          centers=[(0, 0), (0, 2)], 
                          cluster_std=[self.std, self.std], 
                          random_state=self.random_seed)
    