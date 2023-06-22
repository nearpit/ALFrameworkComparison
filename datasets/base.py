import os
import json 

import numpy as np
import requests
from torch.utils.data import Dataset
from sklearn.datasets import load_svmlight_file


class VectoralDataset(Dataset):

    dataset_name = None
    feature_encoder = None
    target_encoder = None
    configs = None

    def __init__(self, split_name):
        super().__init__()

        self.location = "datasets/data/" + self.dataset_name

        if not os.path.exists(self.location):
            os.mkdir(self.location)
    
        self.split_name = split_name

        if not self.file_exists():
            data = self.obtain()
            data = self.split(data)
            data = self.preprocess(data)
            self.save_npy(data)            
        
        self.load_clean()

    def __len__(self):
         return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # return input and output dimensions
    @property
    def dimensions(self):
        return self.x.shape[1], self.y.shape[1]
    
    def file_exists(self):
        splits = ["train.npy", "val.npy", "test.npy"]
        existed_files = os.listdir(self.location)
        return all([elem in existed_files for elem in splits])
    
    def load_clean(self):
        with np.load(f"{self.location}/{self.split_name}.npy") as file:
            self.x = file["x"]
            self.y = file["y"]
    
    def save_npy(self, data_dict):
        for split_name, data in data_dict.items():
            with open(f"{self.location}/{split_name}.npy", "wb") as f:
                np.savez(f, x=data["x"], y=data["y"])   

    def preprocess(self, data):
        for data_shard in data.values():
            x, y = data_shard["x"], np.reshape(data_shard["y"], (-1, self.configs["n_targets"]))
            self.feature_encoder.fit(x)
            self.target_encoder.fit(y)
            
        for shard_name, data_shard in data.items():
            x, y = data_shard["x"], np.reshape(data_shard["y"], (-1, self.configs["n_targets"]))
            data[shard_name]["x"] = self.feature_encoder.transform(x)
            data[shard_name]["y"] = self.target_encoder.transform(y)
        return data

    def _split(self, array_size, shares=[0.6, 0.2]):
        indices = np.arange(array_size)
        idx_to_split = (np.cumsum(shares)*array_size).astype(int)
        permutated_idx = np.random.choice(indices, array_size, replace=False)
        return np.split(permutated_idx, idx_to_split)
     
    def obtain(self):
        pass

    def split(self):
        pass

class SVMDataset(VectoralDataset):
    
    urls_dict = None
    n_features = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def obtain(self):
        data = {}
        for split_name, url in self.urls.items():
            with open(self.location + split_name + "_raw", 'w') as f:
                r = requests.get(url)
                f.writelines(r.content.decode("utf-8"))

            data[split_name] = load_svmlight_file(self.location + split_name + "_raw", 
                                                  n_features=self.configs["n_features"])
            data = self.split(data)
        return data