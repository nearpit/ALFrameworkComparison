from torch.utils.data import Dataset

class VectoralDataset(Dataset):
    def __init__(self, data):
        super().__init__()

    def __len__(self):
         return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    # return input and output dimensions
    @property
    def dimensions(self):
        return self.x.shape[1], self.y.shape[1]
    
    def obtain_raw(self):
        pass

    def prepare(self):
        pass

    def postprocess(self):
        pass