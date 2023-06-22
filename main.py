import numpy as np 
import torch 


import acquisitions
import datasets
from utilities import funcs, MLP, cnst


if __name__ == '__main__':
    args = funcs.get_arguments()
    Dataclass = getattr(datasets, args.dataset.capitalize())
    Acqclass = getattr(acquisitions, args.algorithm.capitalize())


    data = {"train": Dataclass(split_name="train"), 
            "val": Dataclass(split_name="val"),
            "test": Dataclass(split_name="test")}

    np.random.seed(cnst.RANDOM_STATE)
    idx_lb = np.random.choice(Dataclass.configs["pool_size"], size=Dataclass.configs["n_labeled"], replace=False)
    perforamnce = []

    acq_model = Acqclass(clf_arch=MLP,
                         clf_configs=Dataclass.configs["clf"],
                         data=data,
                         idx_lb=idx_lb)
    
    prev_perfomance = 0
    for idx in range(cnst.BUDGET):
        if not len(acq_model.idx_ulb):
            break
        current_performance = acq_model.eval("test")
        print(current_performance, acq_model.eval("val"), idx, len(acq_model.idx_ulb))
        acq_model.train_clf()
        perforamnce.append(current_performance)
        idx_cand = acq_model.query()
        acq_model.update(idx_cand)
        prev_perfomance = current_performance

