from datasets.base import SVMDataset
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

class Splice(SVMDataset):

    urls_dict = {"train":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice_scale",
                 "val_test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice.t"}
    dataset_name = "splice"
    feature_encoder =  MinMaxScaler(),
    target_encoder = FunctionTransformer(lambda x: (x+1)//2) # from {-1, 1} to {0, 1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split(self, data):
        x = data["val_test"]
        return x