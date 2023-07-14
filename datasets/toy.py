import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from datasets.base import VectoralDataset

class Toy(VectoralDataset):

    dataset_name = "toy"
    feature_encoder = MinMaxScaler()
    target_encoder = OneHotEncoder(sparse_output=False)

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)


    def obtain(self):
        non_honeypot_size = self.configs["n_instances"] - self.configs["n_honeypot"]
        x, y = make_moons(n_samples=non_honeypot_size, noise=0.05, random_state=self.random_seed)
        # x_hp, _ = make_blobs(n_samples=self.configs["n_honeypot"], centers=1, cluster_std=[0.2], center_box=(2, 0), random_state=self.random_seed)
        x_hp = self.make_circle(n_samples=self.configs["n_honeypot"], scale_factor=2.5, along_x=0.5, noise=0.15)
        y_hp = np.tile(np.array([0, 1]), int(self.configs["n_honeypot"]/2))
        x = np.concatenate((x, x_hp)).astype(np.float32)
        y = np.concatenate((y, y_hp)).reshape(-1, 1).astype(np.float32)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.savefig(f"{self.location}/dataset_representation.png")
        plt.close()
        return {"x": x, "y":y}

    def split(self, data):
        x, y = data["x"], data["y"]
        train_idx, test_idx = self.conv_split(x.shape[0], shares=[0.8])

        return {"train": {"x":x[train_idx], "y":y[train_idx]}, 
                "test": {"x":x[test_idx], "y":y[test_idx]}}
    
    def make_circle(self, n_samples, scale_factor=1, along_x=0.5, along_y=0.25, noise=None):
        linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        outer_circ_x = along_x + scale_factor*np.cos(linspace_out)
        outer_circ_y = along_y + scale_factor*np.sin(linspace_out)
        generator = np.random.RandomState(self.random_seed)
        x = np.column_stack([outer_circ_x, outer_circ_y])
        if noise is not None:
            x += generator.normal(scale=noise, size=x.shape)
        return x
