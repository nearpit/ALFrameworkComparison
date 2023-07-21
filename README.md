# Active Learning via Reconstruction Models (ALRM)

## New Framework

![framework difference](https://github.com/Nearpit/ALRM/blob/main/plots/flow_diff.png?raw=true)

## Intermediate results
\* Entropy had 25 trials to find the best parameters while others had 50.
![intermediate results](https://github.com/Nearpit/ALRM/blob/main/plots/intermediate_results.png?raw=true)

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

- [X] 50 tuning trials
- [X] Budget ~40 % of the train split
- [X] n initially labeled: [20, 60, 100, all (perfect performance)]
- [X] Validation share: [10%, 50%, 90%]

### \#TODO

- [X] Pytorch seed for splitting as well
- [X] What toy dataset(-s) to use? Moons, Adversarial Moons, Blobs
- [X] Comparison of static hypers and online tuning?
- [X] Diverging Sin Dataset
- [ ] Adjust Keychain naive
- [ ] Check Keychain Steep Level variable
- [ ] Debug Coreset
- [ ] AUC
- [ ] Code keychain non-heuristics (naive + AE output, etc.)
