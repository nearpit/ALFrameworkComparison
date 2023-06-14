from torch.utils.data import Dataset

class VectoralDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.x = data['x']
        self.y = data['y']

    def __len__(self):
         return self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    # return input and output dimensions
    @property
    def dimensions(self):
        return self.x.shape[1], self.y.shape[1]

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("Inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
