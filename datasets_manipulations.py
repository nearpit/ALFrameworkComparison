from os.path import exists
from os import listdir
import requests

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file, make_blobs
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

import utilities.constants as cnst
from utilities.classes import VectoralDataset


def generate_toy():
    current_file = 'datasets/toy/' 
    if not listdir(current_file):
        x, y = make_blobs(n_samples=cnst.N_SAMPLES, centers=2, cluster_std=[0.05, 0.05], center_box=(-0.2, 0.2), random_state=cnst.RANDOM_STATE)
        x_hp, _ = make_blobs(n_samples=cnst.HONEY_POT_SIZE, centers=1, cluster_std=[0.05], center_box=(1, 0.1), random_state=cnst.RANDOM_STATE)
        y_hp = np.random.randint(0, 2, cnst.HONEY_POT_SIZE)
        x = np.concatenate((x, x_hp)).astype(np.float32)
        y = np.concatenate((y, y_hp)).reshape(-1, 1).astype(np.float32)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.savefig('datasets/toy/dataset_representation.png')

        indices = np.arange(x.shape[0])
        train, val, test = np.split(np.random.choice(indices, x.shape[0], replace=False), 
                                         [int(.6*x.shape[0]), int(.8*x.shape[0])])
        for split in ['train', 'val', 'test']:
            with open(f"datasets/toy/{split}.npy", "wb") as f:
                np.savez(f, x=x[eval(split)], y=y[eval(split)])   

        print("Toy Successfully Generated")

def postprocess_svm_data(split_dict, feature_enc, target_enc):
    for split_name, data in split_dict.items():
        x, y = data
        y = y.reshape(-1, 1)
        feature_enc.fit(x)
        target_enc.fit(y)
        
    for split_name, data in split_dict.items():
        x, y = data
        y = y.reshape(-1, 1)
        split_dict[split_name] = (feature_enc.transform(x), target_enc.transform(y))   
    return split_dict



def download_raw_dna(local_path = 'datasets/dna/'):
    
    train_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.tr"
    val_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.val"
    test_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.t"
   

    split_dict = dict()

    for split in ['train', 'val', 'test']:
        current_file = local_path + split + '_raw'
        if not exists(current_file):
            with open(current_file, 'w') as f:
                r = requests.get(eval(f"{split}_url"))
                f.writelines(r.content.decode("utf-8"))

        split_dict[split] = load_svmlight_file(current_file, n_features=180)
    return split_dict


def save_npy(split_dict, local_path = 'datasets/dna/'):
    for split_name, data in split_dict.items():
        current_file = local_path + split_name + '.npy'
        if not exists(current_file):
             with open(current_file, 'wb') as f:
                x, y = data
                np.savez(f, x=x, y=y)
                
def prepare_dna():
    feature_enc = FunctionTransformer(lambda x: x) #any preprocessing is redundant - all features are within {0, 1}
    target_enc = OneHotEncoder(sparse_output=False)
    
    split_dict = download_raw_dna()
    print("DNA Succesfully Downloaded")
    split_dict = postprocess_svm_data(split_dict, feature_enc, target_enc)
    print("DNA Succesfully Preprocessed")

    save_npy(split_dict)

def prepare_datasets():
    prepare_dna()
    generate_toy()
    print('Data preparation is done')


def load_clean_dataset(dataset_name, splits = ['train', 'val', 'test']):
    data_dict = dict()
    for split in splits:
        with np.load(f"datasets/{dataset_name}/{split}.npy") as file:
            data_dict[split] = VectoralDataset({"x":file["x"], "y":file["y"]})
    return data_dict

if __name__=='__main__':
    prepare_datasets()