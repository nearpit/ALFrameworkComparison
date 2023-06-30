from datasets.base import SVMDataset
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class Splice(SVMDataset):
    # split train for validation
    urls_dict = {"train":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice_scale",
                 "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice.t"}
    dataset_name = "splice"
    feature_encoder =  MinMaxScaler()
    target_encoder = OneHotEncoder(sparse_output=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split(self, data, train_share=0.6):
        #CAVEAT: Make sure you have the same train_size in the config file - int(train_share*1000)
        n_instances = len(data["train"]["x"])
        train_idx, val_idx = self.conv_split(n_instances, [train_share])
        data["val"] = {"x": data["train"]["x"][val_idx], "y": data["train"]["y"][val_idx]}
        data["train"] = {"x": data["train"]["x"][train_idx], "y": data["train"]["y"][train_idx]}
        return data