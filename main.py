import numpy as np 

import acquisitions
import datasets
from utilities import funcs, MLP, cnst, EarlyStopper


if __name__ == '__main__':
    args = funcs.get_arguments()
    Dataclass = getattr(datasets, args.dataset.capitalize())
    Acqclass = getattr(acquisitions, args.algorithm.capitalize())

    data = {"train": Dataclass(split_name="train"), 
            "val": Dataclass(split_name="val"),
            "test": Dataclass(split_name="test")}

    np.random.seed(cnst.RANDOM_STATE)
    idx_lb = np.random.choice(Dataclass.configs["train_size"], size=Dataclass.configs["n_labeled"], replace=False)
    perforamnce = []
    
    model_arch_name = "MLP"
    model_configs = {
        "last_activation": getattr(cnst, f"{model_arch_name}_LAST_ACTIVATION"),
        "metrics_dict": Dataclass.configs["clf_metrics"],
        "criterion": getattr(cnst, f"{model_arch_name}_CRITERION"), 
        "batch_size": Dataclass.configs["batch_size"], 
        "optimizer": cnst.OPTIMIZER,
        "early_stop": True
    }
    acq_model = Acqclass(upstream_arch=MLP,
                         upstream_configs=model_configs,
                         data=data,
                         idx_lb=idx_lb)
    
    retuner = EarlyStopper(patience=cnst.HINDERED_ITERATIONS)
    new_hypers = funcs.hypers_search(data=data, model_arch_name="MLP", model_configs=model_configs)
    acq_model.update_upstream_configs(new_hypers)

    for idx in range(Dataclass.configs["train_size"]):

        acq_model.train_upstream()
        test_performance = acq_model.eval("test")
        val_performance = acq_model.eval("val")
        perforamnce.append(test_performance)
        idx_cand = acq_model.query()
        acq_model.update(idx_cand)

        print(val_performance, test_performance, idx_cand, len(acq_model.idx_lb), len(acq_model.idx_ulb))

        if retuner.early_stop(val_performance[0]): # if training is hindered
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW HYPERS WERE REQUESTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            new_hypers = funcs.hypers_search(data=data, model_arch_name="MLP", model_configs=model_configs)
            acq_model.update_upstream_configs(new_hypers)
