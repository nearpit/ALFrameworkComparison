import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from datasets.base import VectoralDataset

class Toy(VectoralDataset):

    feature_encoder = MinMaxScaler()
    target_encoder = OneHotEncoder(sparse_output=False)

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)



    def split(self, data):
        x, y = data["x"], data["y"]
        train_idx, val_idx, test_idx = self.conv_split(x.shape[0])

        return {"train": {"x":x[train_idx], "y":y[train_idx]}, 
                "val" : {"x":x[val_idx], "y": y[val_idx]},
                "test": {"x":x[test_idx], "y":y[test_idx]}}
    
    def obtain(self):
        x, y = self.generate_clean()
        x_hp, _ = self.generate_noise()
        # x_hp = self.make_circle(n_samples=self.configs["n_honeypot"], scale_factor=2.5, along_x=0.5, noise=0.15)
        y_hp = np.tile(np.array([0, 1]), int(self.configs["n_honeypot"]/2))
        x = np.concatenate((x, x_hp)).astype(np.float32)
        y = np.concatenate((y, y_hp)).reshape(-1, 1).astype(np.float32)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.savefig(f"{self.location}/dataset_representation.png")
        plt.close()
        return {"x": x, "y":y}

    def generate_clean(self):
        pass

    def generate_noise(self):
        pass

    def make_circle(self, n_samples, scale_factor=1, along_x=0.5, along_y=0.25, noise=None):
        linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        outer_circ_x = along_x + scale_factor*np.cos(linspace_out)
        outer_circ_y = along_y + scale_factor*np.sin(linspace_out)
        generator = np.random.RandomState(self.random_seed)
        x = np.column_stack([outer_circ_x, outer_circ_y])
        if noise is not None:
            x += generator.normal(scale=noise, size=x.shape)
        return x

class Blobs(Toy):
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
    
class Moons(Toy):
    dataset_name = "moons"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_noise(self):
        return self.make_circle(n_samples=self.configs["n_honeypot"], scale_factor=2.5, noise=0.1), None

    
    def generate_clean(self):
        return make_moons(n_samples=self.configs["n_instances"] - self.configs["n_honeypot"],
                          noise=0.1,
                          random_state=self.random_seed)
        
