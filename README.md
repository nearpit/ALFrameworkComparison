# Active Learning via Reconstruction Models (ALRM)

## New Framework
![framework difference](https://github.com/Nearpit/ALRM/blob/main/overall%20plots/flow_diff.png)

## Installation

```
conda create -n alrm python=3.9 -y
conda activate alrm  
git clone https://github.com/Nearpit/ALRM.git
cd ALRM
pip install .
```

### To add a new acquisition function

1. Add a new class to the _acquisition_ folder inhereting some of the _base_ classes.
2. Add the relative name for the `--a` argument.
3. Import the added class to the _\_\_init\_\_.py_ file in the _acquisition_ folder.

### To add a new dataset

1. Add a new class to the _dataset_ folder inhereting some of the _base_ classes.
2. Import the added class to the _\_\_init\_\_.py_ file in the _acquisition_ folder.
3. Initialize json file in the _dataset/configs_ folder (see example.json)
   1. __n_features__ - the number of features of the added dataset
   2. __n_instances__ - the number of instances of the added dataset
   3. __n_labeled__ - the number of labeled instances at the begging of the AL process
   4. __train_size__ - the number of instances in the train set(i.e. number of labeled  + unlabeled instances)
   5. __n_classes__ - the number of classes of the target space
   6. __batch_size__ (x < n_labeled)
   7. __budget__ - the amount of AL iterations (0 < x < train_size - n_labeled)
   8. __metrics_dict__ - metrics of the classifier

## Findings

- Conventionally used static hyperparameters lead to irrelevant classfier. There is a clear tendecy of drastically deteriorating the classifier once in a while.
- Due to the previous point, BALD suffered the most - its acquisition function looks like noise most of the time.

## Online Tuning

- [X] ~~50~~ 25 tuning trials
- [X] Budget ~40 % of the train split
- [ ] Initially labeled:
  - [X] 20
  - [ ] 60
  - [ ] 100
  - [X] all (perfect performance)
- [ ] Validation share
  - [ ] 10%
  - [X] 50%
  - [ ] 90%

### \#TODO

- [X] Adjust Keychain naive
- [X] Debug Coreset
- [X] Remove log(loss) in visualization
- [X] Query density plots
  - [X] added_x, added_y
  - [X] added_idx
- [X] Debug dataset budget config
- [X] Keychain Autoencoder
- [ ] AUC?

## Further steps after the master's thesis

- [ ] Try out above-mentioned online tuning circumstances
- [ ] Find how to make replay buffer work
- [ ] Batch-wise keychain
