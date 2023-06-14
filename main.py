from datasets_manipulations import prepare_datasets, load_clean_dataset
from utilities import constants as cnst
from utilities import funcs
import numpy as np 

if __name__ == '__main__':
    prepare_datasets()
    args = funcs.get_arguments()

    data = load_clean_dataset(dataset_name=args.dataset)
    train, val, test = data['train'], data['val'], data['test']

    labeled_indices = np.random.choice(np.arange(train['y'].shape[0]), size=cnst.INITIAL_LABELED, replace=False)
    unlabeled_indices = np.delete(np.arange(train['y'].shape[0]), labeled_indices)