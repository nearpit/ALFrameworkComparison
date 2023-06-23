# Active Learning via Reconstruction Models (ALRM)
## Installation

` conda create -n alrm python=3.9 &nbsp
conda activate alrm  
git clone https://github.com/Nearpit/ALRM.git
pip install .`

### To add a new acquisition function

1. Add a new class to the _acquisition_ folder inhereting some of the _base_ classes.
2. Add the relative name for the `--a` argument.
3. Import the added entity to the _\_\_init\_\_.py_ file in the _acquisition_ folder.

### To add a new dataset

1. Add a new class to the _dataset_ folder inhereting some of the _base_ classes.
2. Add the relative name for the `--d` argument.
3. Import the added entity to the _\_\_init\_\_.py_ file in the _acquisition_ folder.


### \#TODO
- [ ] Debug DNA
- [ ] Debug Splice
- [ ] Finish ALRM 
