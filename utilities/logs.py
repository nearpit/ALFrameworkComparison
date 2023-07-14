import pickle
import os
import utilities

def gather_results(args, last_cand, test_perf, val_perf, pool, idx):
        return {
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "random_seed": args.random_seed,
            "last_cand": last_cand,
            "test_loss": test_perf[0],
            "test_acc": test_perf[1]["MulticlassAccuracy"],
            "val_loss": val_perf[0],
            "val_acc": val_perf[1]["MulticlassAccuracy"],
            "len_ulb": pool.get_len("unlabeled"),
            "len_lb": pool.get_len("labeled"), 
            "iter": idx
        }
    
def get_name(args, include_alg=True):
    if include_alg:
        return f"{args.dataset}_{args.algorithm}_{args.random_seed}"
    else:
        return f"{args.dataset}_{args.random_seed}"

def store_file(array, filename, path="results/"):
    utilities.makedir(path)
    with open(f'{path}{filename}', 'wb') as file:
        pickle.dump(array, file)

def retrieve_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            x = pickle.load(file)
    else:
        x = None
    return x