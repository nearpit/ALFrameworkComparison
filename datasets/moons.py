from sklearn.datasets import make_moons
from datasets.toy import Toy

class Moons(Toy):
    dataset_name = "moons"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_noise(self):
        return self.make_circle(n_samples=self.configs["n_honeypot"], scale_factor=2.5, noise=0.1), None

    
    def generate_clean(self):
        return make_moons(n_samples=self.configs["n_instances"] - self.configs["n_honeypot"],
                          noise=0.075,
                          random_state=self.random_seed)