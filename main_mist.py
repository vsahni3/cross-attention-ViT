from dataset_ucsf import BrainDataset, clean_data
from model_cross import ModelCross
from modelv3 import ModelVIT
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data.distributed import DistributedSampler 
import config2 
import config
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from collections import namedtuple
from sklearn.model_selection import KFold, StratifiedKFold
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# import os
# from utils import compute_metrics
# import numpy as np
# from scipy.stats import ttest_rel
# import torchmetrics


# DIM / IMAGE SIZE ONLY 128 MAYBE UPSCALE IF NEEDED
# Load configuration



file_path = '/scratch/p/ptyrrell/vsahni3'
# Callbacks


# early_stop_callback = EarlyStopping(
#     monitor="val_loss",  # metric name to monitor
#     min_delta=0.00,      # minimum change in the monitored metric to be considered an improvement
#     patience=25,          # number of epochs with no improvement after which training will be stopped
#     verbose=True,        # whether to print messages when stopping early
#     mode="min"           # mode "min" because we want the loss to be as low as possible
# )

def create_sampler(train_df, target):
    num_negative = len(train_df[train_df[target] == 0])
    num_positive = len(train_df) - num_negative

    # neg must be first cuz of indexing by label
    class_counts = [num_negative, num_positive]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[int(label)] for label in train_df[target]]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler



# Instantiate the model
# model.apply(reset_weights)
Params = namedtuple("Params", ["lr", "dropout", "attn_order", "optim_params", "weight_decay", "img_types", "label_smoothing", "img_aug"])
# label smoothing
# stochatsic depth
# precision affects
# shuffle labels to see if truly memorizng
#AJWIDNWEFNIEFNEOJFKEFMEMFE
mods = ['DWI', 'SWI', 'T1c', 'brain_parenchyma_segmentation', 'tumor_segmentation', 'T2', 'ADC', 'ASL', 'FLAIR']
mods_o = ['DTI_eddy_L3', 'DTI_eddy_FA', 'DTI_eddy_L1', 'DTI_eddy_L2', 'DTI_eddy_MD', 'DWI_bias', 'SWI_bias', 'T1c_bias']
params_list1 = [
    # have to use str for attn_order otherwise config throws error when setting keys
    Params(lr=1e-4, dropout=0.2, attn_order={'0': '1', '1': '2', '2': '0'}, optim_params={"T_max": 250, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[0], mods[1], mods[7]), label_smoothing=0.0, img_aug=True),
    Params(lr=1e-4, dropout=0.2, attn_order={'0': '1', '1': '2', '2': '0'}, optim_params={"T_max": 200, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[0], mods[1], mods[7]), label_smoothing=0.0, img_aug=True),
    ]

params_list2 = [
    Params(lr=1e-4, dropout=0.1, attn_order={}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0, img_aug=False),
    Params(lr=1e-4, dropout=0.1, attn_order={}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0, img_aug=True),
    
]




def train_cv():
    # run this with fixed test seed
    run = 145
    models = [ModelCross, ModelVIT]
    configs = [config2, config]
        
    big_data = pd.read_csv("labels.csv")
    
    seeds1 = [6253, 9253]
    seeds_val = [[6253, 9253], [6253, 9253]]
    data = clean_data(data, config.target)
    for r in range(len(seeds3)):
        
        for m in range(2):
            data, test_df = train_test_split(data, test_size=0.15, random_state=seeds1[r])
            config_file = configs[m]
            cur_config = config_file.get_mgmt_config()
            
            model_bp = models[m]
            k = 5
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seeds_val[m][r])
            # kfold = KFold(n_splits=k, shuffle=True, random_state=909)
            for i, params in enumerate(params_big[m]):

                for fold, (train_idx, val_idx) in enumerate(kfold.split(data, data[cur_config.target])):
                    
                    
                    
                    lightning_logger = TensorBoardLogger(save_dir=f"{file_path}/lightning_logs/cross", name=f"{run}_{i}_{fold}_{m}_{r}")
                    csv_logger = CSVLogger(save_dir=f"{file_path}/csv_logs/cross", name=f"{run}_{i}_{fold}_{m}_{r}")

                    cur_config = config_file.modify_config(cur_config, params)
                    cur_config = config_file.modify_config(cur_config, {'num_modalities': len(params.img_types)})
                    model = model_bp(cur_config)
                    

                    train_df = data.iloc[train_idx]
                    # does poorly when balanced val
                    val_df = data.iloc[val_idx]


                    sampler = create_sampler(train_df, cur_config.target)



                    train_dataset = BrainDataset(config=cur_config, data=train_df, is_train=True, types=params.img_types)
                
                    val_dataset = BrainDataset(config=cur_config, data=val_df, is_train=False, types=params.img_types)

                



                    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=5, sampler=sampler)
                    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=5)


                    torch.cuda.empty_cache()
                    trainer = L.Trainer(
                    max_epochs=250,
                    accelerator="auto",
                    logger=[lightning_logger, csv_logger],
                    devices=4,
                    num_nodes=2,
                    )
                    trainer.fit(model, train_loader, val_loader)






def train_full(params_big):
    run = 185
    models = [ModelCross, ModelVIT]
    configs = [config2, config]
        
    big_data = pd.read_csv("labels.csv")
    
    test_seeds = [2004, 1969, 911, 100]
    big_data = clean_data(big_data, "MGMT status")
    
    for r in range(len(test_seeds)):
        data, test_df = train_test_split(big_data, test_size=0.15, random_state=test_seeds[r])
        for m in range(2):
            config_file = configs[m]
            cur_config = config_file.get_mgmt_config()
            
            model_bp = models[m]
            for i, params in enumerate(params_big[m]):
                checkpoint_callback = ModelCheckpoint(
                    dirpath=f"{file_path}/checkpoints/cross",           
                    monitor="val_auc_roc",          
                    filename="{epoch:02d}-{val_auc_roc:.4f}" + f'test_{run}_{r}_{m}_{i}', 
                    save_top_k=5,                   
                    mode="max",                      
                )
                # .18 * .85 ~ 0.15
                train_df, val_df = train_test_split(data, test_size=0.18, random_state=test_seeds[r])
                lightning_logger = TensorBoardLogger(save_dir=f"{file_path}/lightning_logs/cross", name=f'test_{run}_{r}_{m}_{i}')
                csv_logger = CSVLogger(save_dir=f"{file_path}/csv_logs/cross", name=f'test_{run}_{r}_{m}_{i}')

                cur_config = config_file.modify_config(cur_config, params)

                cur_config = config_file.modify_config(cur_config, {'num_modalities': len(params.img_types)})

                model = model_bp(cur_config)



                sampler = create_sampler(train_df, cur_config.target)



                train_dataset = BrainDataset(config=cur_config, data=train_df, is_train=True, types=params.img_types)
            
                val_dataset = BrainDataset(config=cur_config, data=val_df, is_train=False, types=params.img_types)

            



                train_loader = DataLoader(train_dataset, batch_size=8, num_workers=5, sampler=sampler)
                val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=5)


                torch.cuda.empty_cache()
                trainer = L.Trainer(
                max_epochs=250,
                accelerator="auto",
                logger=[lightning_logger, csv_logger],
                devices=4,
                num_nodes=2,
                callbacks=[checkpoint_callback]
                )
                trainer.fit(model, train_loader, val_loader)
                    
train_full([params_list1, params_list2])
# # use same seed as above                    
