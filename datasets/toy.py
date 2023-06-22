import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from datasets.base import VectoralDataset
import utilities.constants as cnst

class Toy(VectoralDataset):

    dataset_name = "toy"
    feature_encoder = MinMaxScaler()
    target_encoder = FunctionTransformer(lambda x: x) # identity transformation

    with open(f"datasets/configs/{dataset_name}.json", "r") as openfile:
        configs = json.load(openfile)
    

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)


    def obtain(self):
        non_honeypot_size = self.configs["n_instances"] - self.configs["n_honeypot"]
        x, y = make_blobs(n_samples=non_honeypot_size, centers=2, cluster_std=[0.05, 0.05], center_box=(-0.2, 0.2), random_state=cnst.RANDOM_STATE)
        x_hp, _ = make_blobs(n_samples=self.configs["n_honeypot"], centers=1, cluster_std=[0.05], center_box=(1, 0.1), random_state=cnst.RANDOM_STATE)
        y_hp = np.random.randint(0, 2, self.configs["n_honeypot"])
        x = np.concatenate((x, x_hp)).astype(np.float32)
        y = np.concatenate((y, y_hp)).reshape(-1, 1).astype(np.float32)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.savefig(f"{self.location}/dataset_representation.png")
        return {"x": x, "y":y}

    def split(self, data):
        x, y = data["x"], data["y"]
        train_idx, val_idx, test_idx = self._split(x.shape[0])

        return {"train": {"x":x[train_idx], "y":y[train_idx]}, 
                "val": {"x":x[val_idx], "y":y[val_idx]},
                "test": {"x":x[test_idx], "y":y[test_idx]}}
    
