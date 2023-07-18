# Active Learning via Reconstruction Models (ALRM)

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
2. Add the relative name for the `--d` argument.
3. Import the added class to the _\_\_init\_\_.py_ file in the _acquisition_ folder.
4. Initialize json file in the _dataset/configs_ folder (see example.json)
   1. __n_features__ - the number of features of the added dataset
   2. __n_instances__ - the number of instances of the added dataset
   3. __n_labeled__ - the number of labeled instances at the begging of the AL process
   4. __train_size__ - the number of instances in the train set(i.e. number of labeled  + unlabeled instances)
   5. __n_classes__ - the number of classes of the target space
   6. __batch_size__ (x < n_labeled)
   7. __budget__ - the amount of AL iterations (0 < x < train_size - n_labeled)
   8. __metrics_dict__ - metrics of the classifier

### \#TODO
- [ ] Keychain naive
- [ ] Keychain non-heuristics (naive + AE output, etc.)
- [ ] Keychain __SUPERIOR__ (Heuristics + non-heuristics)
- [ ] Uncertainty Quantification via Autoencoder(AE)

### What is hapenning?

1. Still staying with online tuning (the conventional framework is also implemented and can be switched to any time). They can be seen on gifs. The online tuning seems more legit in the real world scenario since we don't have the access to the whole dataset at the beginning.
2. All aquisition functions tend to pick predominantly noise in the online framework (without perfect hyper parameters that were found on the whole dataset). I presume that it might be the reason AL is not working in the real world scenario.
3. Split. When I tried to tune the hypers every iteration, there was a huge disproportion between train/validation splits (e.g. 10 vs 500) what hindered the training process. I presume Repeated K-Fold (2 folds and many repeatitions) is the way to go for finding the best hyper parameters for the current iteration. 2 folds were chosen to alleviate picking noisy instances.
4. DRUMROLL! All mentioned above still do NOT lead to the working prototype :C (show the last experiment). The learning process seems overregularized.

Regularization:
L2, early stopping is off for now.

Tuning:
Ranges were expanded.
Considering LR scheduler to be added while LR set to 0.5 and set staticly the architecture to the arbitarly sufficient one (or simplify to the depth and width).  
