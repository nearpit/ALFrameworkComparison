import os 
import datasets

if __name__ == "__main__":
    location  = "datasets/configs/"
    datasets_list = os.listdir(location)
    datasets_list = [x[:-5] for x in datasets_list] # remove .json from the right
    datasets_list.remove("example") # remove example file
    for dataset_name in datasets_list:
        Dataclass = getattr(datasets, dataset_name.capitalize())
        _ = Dataclass("train") #download and initialize 