# Identifying Visually Meaningless Prototypes in PIP-Net

In this code folder we have removed runs, results, models, etc. in order to save space. The files in this folder allow to reproduce the steps of our research. 

This folder also includes original code of PIP-Net repository. However, some parts were modified in order to execture our experiments

## Steps to reproduce 

1. **Data.** First create ./data/PLK_Mini/ folder next to main.py. To download and split the PLK_Mini dataset use plankton_dataset_load.ipynb notebook.
2. To train the model run command `python3 main.py --dataset plankton --image_size 128 --epochs_pretrain 10 --batch_size 16 --freeze_epochs 0 --epochs 60 --log_dir ./runs/plankton`
3. To evaluate the model and generate the final matrix run `python3 main.py --dataset plankton --image_size 128 --epochs_pretrain 0 --batch_size 16 --freeze_epochs 0 --epochs 0 --log_dir ./runs/plankton --state_dict_dir_net ./runs/net_trained_last` 

Commands for training and evaluation are also present in plankton_dataset_load.ipynb for both 258 and 768 prototypes. After training and evaluation the notebooks for Bi-Clustering, Beta prunning and Recursive Feature Elimination can be executed. Keep in mind that the names of the matrices files are needed to be changed according to the newly generated files where we save the similarity score matrices from  evaluation tuns

## Mentions 

* `proto_drop_vis.py` is a file used for visualizing dropped and kept prototypes for evaluating algorithms 
