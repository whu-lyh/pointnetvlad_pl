
# Installation
```
cd scripts
sh install.sh
```
> `docker` is recommanded.
>  the pytorch_lightning is the newest version

# Dataset Round 1
Trainset
```
cat TRAIN.tar.gz.* | tar -zxv
```
the data folder is as follows:
```
.
├── pretrained # generated during the dataset init func is called at first time 
├── queries.pickle # generated during the dataset init func is called at first time 
├── train_1
├────── 000001.pcd
├────── 000001_pose6d.npy
├──────...
├────── 000001093.pcd
├────── 000001093_pose6d.npy
├── ...
├── train_15
```

validation data folder example.
```
VAL/
VAL/DATABASE/
VAL/DATABASE/val_4/
VAL/DATABASE/val_4/000025.pcd
VAL/DATABASE/val_4/000025_pose6d.npy
```

+ the submap is under global coordinate, thus the pose is implicitly embeded inside it.

# Train
```
python train.py --root_dir GPR_competition/UGV/TRAIN
```
> When run this scripts first time, the pretrained file is generated and quite time-consuming.
# Test
