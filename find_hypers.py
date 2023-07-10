import core, utilities, datasets
import logging

def find_clf_hypers(args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        data = Dataclass.get_data_dict()
        pool = core.Pool(data=data, random_seed=args.random_seed, whole_dataset=True)
        clf = core.Learnable(pool=pool, random_seed=args.random_seed)
        clf.tune_model()
        return clf.model_configs.copy(), clf.eval_model("test")

if __name__ == "__main__":
    args = utilities.get_arguments()
    hypers, best_perf = find_clf_hypers(args)
    path_to_store = "results/aux/"
    filename = utilities.get_name(args, include_alg=False)
    print(hypers, best_perf)
    utilities.store_results(hypers, filename=filename + "_best_hypers", path=path_to_store)
    utilities.store_results(best_perf, filename=filename + "_best_performance", path=path_to_store)
    logging.warning(f"{hypers} {best_perf}")
    logging.warning("!"*50 + "DONE" + "!"*50)
