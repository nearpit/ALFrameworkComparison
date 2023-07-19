from sklearn.datasets import make_blobs

from datasets.toy import Toy

class Div_sin(Toy):
    dataset_name = "blobs"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_noise(self):
        return make_blobs(n_samples=self.configs["n_honeypot"], 
                          centers=[(0, 1.5)],
                          cluster_std=[0.3],
                          random_state=self.random_seed)

    
    def generate_clean(self):
        return make_blobs(n_samples=self.configs["n_instances"] - self.configs["n_honeypot"],
                          centers=[(0, 0), (1, 2)], 
                          cluster_std=[0.2, 0.2], 
                          random_state=self.random_seed)
    