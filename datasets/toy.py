import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from datasets.base import VectoralDataset

class Toy(VectoralDataset):
    visualize = True
    feature_encoder = MinMaxScaler()
    target_encoder = OneHotEncoder(sparse_output=False)


    def __init__(self, *args, **kwargs):
        self.generator = np.random.RandomState(self.random_seed)
        super().__init__(*args, **kwargs)




    def split(self, data):
        x, y = data["x"], data["y"]
        train_idx, test_idx = self.conv_split(x.shape[0], shares=[0.8])

        return {"train": {"x":x[train_idx], "y":y[train_idx]}, 
                "test": {"x":x[test_idx], "y":y[test_idx]}}
    
    def obtain(self):
        x, y = self.generate_clean()
        x_hp, _ = self.generate_noise()
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
        x = np.column_stack([outer_circ_x, outer_circ_y])
        if noise is not None:
            x += self.generator.normal(scale=noise, size=x.shape)
        return x