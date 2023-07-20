import os
import json 

import numpy as np
import requests
from torch.utils.data import Dataset
from sklearn.datasets import load_svmlight_file

class VectoralDataset(Dataset):
    visualize = False
    dataset_name = None
    feature_encoder = None
    target_encoder = None
    configs = None
    random_seed = 42

    # Loading configs for every non-base dataset class
    def __init_subclass__(cls, **kwargs):
        if cls.dataset_name:
            with open(f"datasets/configs/{cls.dataset_name}.json", "r") as openfile:
                cls.configs = json.load(openfile)
        super().__init_subclass__(**kwargs)


    def __init__(self, split_name):
        super().__init__()

        self.location = "datasets/data/" + self.dataset_name

        if not os.path.exists(self.location):
            os.makedirs(self.location)
    
        self.split_name = split_name

        if not self.file_exists():
            data = self.obtain()
            data = self.split(data)
            data = self.preprocess(data)
            self.save_npz(data)            
        
        self.load_clean()

    def __len__(self):
         return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
  
    def file_exists(self):
        splits = ["train.npz", "test.npz"]
        existed_files = os.listdir(self.location)
        return all([elem in existed_files for elem in splits])
    
    def load_clean(self):
        with np.load(f"{self.location}/{self.split_name}.npz", allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"].astype(np.float32)
    
    def save_npz(self, data_dict):
        for split_name, data in data_dict.items():
            with open(f"{self.location}/{split_name}.npz", "wb") as f:
                np.savez(f, x=data["x"], y=data["y"])   

    def preprocess(self, data):
        for data_shard in data.values():
            x, y = data_shard["x"], data_shard["y"]
            self.feature_encoder.fit(x)
            self.target_encoder.fit(y)
            
        for shard_name, data_shard in data.items():
            x, y = data_shard["x"], data_shard["y"]
            data[shard_name]["x"] = self.feature_encoder.transform(x)
            data[shard_name]["y"] = self.target_encoder.transform(y)
           
        return data
    
    @classmethod
    def get_data_dict(cls):
        return {"train": cls(split_name="train"), 
                "test": cls(split_name="test")}
    
    @staticmethod
    def conv_split(array_size, shares=[0.6, 0.2], seed=42):
        indices = np.arange(array_size)
        idx_to_split = (np.cumsum(shares)*array_size).astype(int)
        np.random.seed(seed)
        permutated_idx = np.random.choice(indices, array_size, replace=False)
        return np.split(permutated_idx, idx_to_split)

    @staticmethod
    def step_split(array_size, val_chunk):
        indices = np.arange(array_size)
        train_idx = np.random.choice(indices[:-val_chunk], indices[:-val_chunk].shape[0], replace=False)
        val_idx = indices[-val_chunk:]
        return train_idx, val_idx
     
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
        for split_name, url in self.urls_dict.items():
            file_path = f"{self.location}_{split_name}_raw"
            with open(file_path, 'w') as f:
                r = requests.get(url)
                f.writelines(r.content.decode("utf-8"))
            x, y  = load_svmlight_file(file_path, n_features=self.configs["n_features"])
            data[split_name] = {"x": np.asarray(x.todense(), dtype=np.float32), "y": y.reshape(-1, 1)}
            os.remove(file_path)
        
        return data
    
class ReplayDataset(Dataset):

    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.configs = {
            "train_size": len(x),
            "n_labeled" : len(x),
            "batch_size": 32,
            "metrics_dict": {},
            "n_features": x.shape[1],
            "n_classes": y.shape[1]
        }
   
    def __len__(self):
         return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]