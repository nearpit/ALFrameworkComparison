# Name Master Thesis

## Acquisition Functions Visualization

### Random Sampling Baseline

![random](https://github.com/Nearpit/ALRM/blob/main/visualization/entropy.gif)

### Coreset

![coreset](https://github.com/Nearpit/ALRM/blob/main/visualization/coreset.gif)

### Entropy

![entropy](https://github.com/Nearpit/ALRM/blob/main/visualization/entropy.gif)

### BALD

![bald](https://github.com/Nearpit/ALRM/blob/main/visualization/bald.gif)

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
   3. __n_classes__ - the number of classes of the target space
   4. __batch_size__ - training batch size
   5. __budget__ - the amount of AL iterations (0 < x < train_size - n_labeled)
   6. __metrics_dict__ - metrics of the classifier
