import numpy as np 

import acquisitions
import datasets
from utilities import funcs, NN, cnst, EarlyStopper


if __name__ == '__main__':
    args = funcs.get_arguments()
    Dataclass = getattr(datasets, args.dataset.capitalize())
    Acqclass = getattr(acquisitions, args.algorithm.capitalize())

    data = {"train": Dataclass(split_name="train"), 
            "val": Dataclass(split_name="val"),
            "test": Dataclass(split_name="test")}

    np.random.seed(cnst.RANDOM_STATE)
    idx_lb = np.random.choice(Dataclass.configs["train_size"], size=Dataclass.configs["n_labeled"], replace=False)
    performance = []

    acq_model = Acqclass(data=data, idx_lb=idx_lb)
    
    retuner = EarlyStopper(patience=cnst.HINDERED_ITERATIONS)

    new_hypers = acq_model.tuner()
    acq_model.update_model_configs(new_hypers)

    for idx in range(Dataclass.configs["budget"]):

        acq_model.train_model()
        test_performance = acq_model.eval_model("test")
        val_performance = acq_model.eval_model("val")
        performance.append(test_performance)
        idx_cand = acq_model.query()
        acq_model.add_new_inst(idx_cand)

        print(val_performance, test_performance, idx_cand, len(acq_model.idx_lb), len(acq_model.idx_ulb))

        if retuner.early_stop(val_performance[0]): # if training is hindered
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW UPSTREAM HYPERS WERE REQUESTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            new_hypers = acq_model.tuner()
            acq_model.update_model_configs(new_hypers)
