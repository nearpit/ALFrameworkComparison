import numpy as np
from sklearn.datasets import make_blobs
from datasets.toy import Toy

class Div_sin(Toy):
    dataset_name = "div_sin"
    sin_freq = 2
    divergence_factor = 0.3
    curve_distance=0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

   
    def generate_clean(self, noise=0.5):
        n_samples=int((self.configs["n_instances"] - self.configs["n_honeypot"])/2)
        x = np.linspace(0, 10, n_samples)
        sin_curve = np.sin(self.sin_freq*x)

        # Cluster above the curve
        cluster_above_x = x
        cluster_above_y = sin_curve + self.divergence_factor * x + self.generator.normal(scale=noise, size=x.shape)
        cluster_above = np.c_[cluster_above_x, cluster_above_y]


        # Cluster below the curve
        cluster_below_x = x
        cluster_below_y = (sin_curve - self.divergence_factor * x + self.generator.normal(scale=noise, size=x.shape)) - self.curve_distance
        cluster_below = np.c_[cluster_below_x, cluster_below_y]


        x = np.concatenate([cluster_above, cluster_below])
        y = np.concatenate([np.zeros(len(cluster_below_y)), np.ones(len(cluster_above_y))])
        return x, y
    
    def generate_noise(self):
        return np.empty((0, self.configs["n_features"])), None