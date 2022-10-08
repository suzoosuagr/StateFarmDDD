# Leave subjects out  StateFarm Dataset for Distracted Behavior detection. 

The dataset is available from : https://www.kaggle.com/c/state-farm-distracted-driver-detection 

### packages:
- termcolor
- pycm
- timm
- torch


### Split
This project only used the training splits of this dataset with k-fold cross validation style. each folder includes 3 subjects. 

### Training

run `~/Scripts/train_state_farm.sh`
```bash
$ cd /StateFarmDDD
$ . ./Scripts/train_state_farm.sh
```
