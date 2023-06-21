import numpy as np 
import torch 

from datasets_manipulations import prepare_datasets, load_clean_dataset
from utilities import constants as cnst
from utilities import funcs
from utilities.models import MLP
import acquisitions

import matplotlib.pyplot as plt


if __name__ == '__main__':
    prepare_datasets()
    args = funcs.get_arguments()
    configs = funcs.get_configs(args.dataset)

    data = load_clean_dataset(dataset_name=args.dataset)
    train, val, test = data['train'], data['val'], data['test']

    np.random.seed(cnst.RANDOM_STATE)

    idx_lb = np.random.choice(np.arange(train.y.shape[0]), size=cnst.INITIAL_LABELED, replace=False)

    perforamnce = []
    model_arch = MLP
    model_configs = configs["clf"]
    acq_model = getattr(acquisitions, args.algorithm.capitalize())(model_arch=model_arch,
                                                                   model_configs=model_configs,
                                                                   data=data,
                                                                   idx_lb=idx_lb)
    prev_perfomance = 0
    for idx in range(cnst.BUDGET):
        if not len(acq_model.idx_ulb):
            break
        current_performance = acq_model.eval("test")
        # if prev_perfomance != current_performance or idx % 20 == 0:
        #     x, y = acq_model.get_labeled()
        #     fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        #     ax.scatter(x[:, 0], x[:, 1], c=y)
        #     x, y = acq_model.get_unlabeled()
        #     ax.scatter(x[:, 0], x[:, 1], c='grey')
        #     title = "Sanity check" if idx % 20 ==0 else f"Performance Changed {idx}"
        #     ax.set_title(title)
        #     plt.show()
        #     print('!!!!!!!!!!!!!!!!!!!!')

        print(current_performance, acq_model.eval("val"), idx, len(acq_model.idx_ulb))
        acq_model.train_clf()
        perforamnce.append(current_performance)
        idx_cand = acq_model.query()
        acq_model.update(idx_cand)
        prev_perfomance = current_performance

