from datasets.base import SVMDataset
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class Splice(SVMDataset):
    # split train for validation
    urls_dict = {"train":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice_scale",
                 "val_test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice.t"}
    dataset_name = "splice"
    feature_encoder =  MinMaxScaler(),
    target_encoder = OneHotEncoder(sparse_output=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split(self, data):
        x = data["val_test"]
        return x