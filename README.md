# Active Learning via Reconstruction Models (ALRM)
## Installation

```
conda create -n alrm python=3.9
conda activate alrm  
git clone https://github.com/Nearpit/ALRM.git
cd ALRM
pip install .
```



### To add a new acquisition function

1. Add a new class to the _acquisition_ folder inhereting some of the _base_ classes.
2. Add the relative name for the `--a` argument.
3. Import the added entity to the _\_\_init\_\_.py_ file in the _acquisition_ folder.

### To add a new dataset

1. Add a new class to the _dataset_ folder inhereting some of the _base_ classes.
2. Add the relative name for the `--d` argument.
3. Import the added entity to the _\_\_init\_\_.py_ file in the _acquisition_ folder.
4. Initialize json file in the _dataset/configs_ folder (see example.json)
   1. __n_features__ - the number of features of the added dataset
   2. __n_instances__ - the number of instances of the added dataset
   3. __n_labeled__ - the number of labeled instances at the begging of the AL process
   4. __train_size__ - the number of instances in the train set(i.e. number of labeled  + unlabeled instances)
   5. __n_classes__ - the number of classes of the target space (CAVEAT, set it to 1 in case of binary classification)


### \#TODO
- [x] Refactor binary classification to cross-entropy
- [x] Finish Online Training/Tuning
- [x] Debug DNA
- [x] Debug Splice
- [ ] Finish ALRM 
