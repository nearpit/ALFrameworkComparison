import numpy as np 

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

    idx_lb = np.random.choice(np.arange(train.y.shape[0]), size=cnst.INITIAL_LABELED, replace=False)

    # train_loader = torch.utils.data.DataLoader(train, batch_size=configs['batch_size'], shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val, batch_size=configs['batch_size'])
    # test_loader = torch.utils.data.DataLoader(test, batch_size=configs['batch_size'])   
    perforamnce = []
    model = MLP(**configs["clf"])
    acq_model = acquisitions.RandomSampler(model, data, idx_lb)
    prev_perfomance = 0
    for idx in range(cnst.BUDGET):
        if not len(acq_model.idx_ulb):
            break
        current_performance = acq_model.eval("test")



        if prev_perfomance != current_performance or idx % 20 == 0:
            x, y = acq_model.get_labeled()
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            ax.scatter(x[:, 0], x[:, 1], c=y)
            x, y = acq_model.get_unlabeled()
            ax.scatter(x[:, 0], x[:, 1], c='grey')
            title = "Sanity check" if idx % 20 ==0 else f"Performance Changed {idx}"
            ax.set_title(title)
            plt.show()
            print('!!!!!!!!!!!!!!!!!!!!')

        print(current_performance, acq_model.eval("val"), idx, len(acq_model.idx_ulb))
        acq_model.train_clf()
        perforamnce.append(current_performance)
        idx_cand = acq_model.query()
        acq_model.update(idx_cand)
        prev_perfomance = current_performance

