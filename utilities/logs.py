import pickle
import os
import utilities
import csv

def gather_results(args, idx_added, features_added, target_added, test_perf, val_perf, train_perf, pool, idx):
        return [args.dataset, 
                args.algorithm, 
                args.random_seed, 
                idx_added, 
                features_added,
                target_added,
                test_perf[0], 
                test_perf[1]["MulticlassAccuracy"], 
                val_perf[0], 
                val_perf[1]["MulticlassAccuracy"], 
                train_perf[0],
                train_perf[1]["MulticlassAccuracy"], 
                pool.get_len("unlabeled"), 
                pool.get_len("all_labeled"), 
                args.val_share,
                args.online,
                args.n_initially_labeled,
                idx]
    
def get_name(args, include_alg=True):
    if include_alg:
        return f"{args.dataset}_{args.algorithm}_{args.random_seed}"
    else:
        return f"{args.dataset}_{args.random_seed}"

def store_pkl(array, filename, path="results/"):
    utilities.makedir(path)
    with open(f'{path}{filename}', 'wb') as file:
        pickle.dump(array, file)

def store_csv(array, filename, path="results/"):
    utilities.makedir(path)
    with open(f'{path}{filename}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)
        
def retrieve_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            x = pickle.load(file)
    else:
        x = None
    return x

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)