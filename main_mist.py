from dataset_ucsf import BrainDataset, clean_data
from model_cross import Model
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data.distributed import DistributedSampler 
from config2 import modify_config, get_mgmt_config
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from collections import namedtuple
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os



# DIM / IMAGE SIZE ONLY 128 MAYBE UPSCALE IF NEEDED
# Load configuration
config = get_mgmt_config()


file_path = '/scratch/p/ptyrrell/vsahni3'
# Callbacks


# early_stop_callback = EarlyStopping(
#     monitor="val_loss",  # metric name to monitor
#     min_delta=0.00,      # minimum change in the monitored metric to be considered an improvement
#     patience=25,          # number of epochs with no improvement after which training will be stopped
#     verbose=True,        # whether to print messages when stopping early
#     mode="min"           # mode "min" because we want the loss to be as low as possible
# )






def create_sampler(train_df):
    num_negative = len(train_df[train_df[config.target] == 0])
    num_positive = len(train_df) - num_negative

    # neg must be first cuz of indexing by label
    class_counts = [num_negative, num_positive]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[int(label)] for label in train_df[config.target]]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


# Instantiate the model

# model.apply(reset_weights)
Params = namedtuple("Params", ["lr", "dropout", "attn_order", "optim_params", "weight_decay", "img_types", "label_smoothing"])
# label smoothing
# stochatsic depth
# precision affects
# shuffle labels to see if truly memorizng
#AJWIDNWEFNIEFNEOJFKEFMEMFE
mods = ['DWI', 'SWI', 'T1c', 'brain_parenchyma_segmentation', 'tumor_segmentation', 'T2', 'ADC', 'ASL']
params_list = [
    Params(lr=1e-4, dropout=0.1, attn_order={0: 1, 1: 0}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0),
    Params(lr=1e-4, dropout=0.1, attn_order={0: 1}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0),
    Params(lr=1e-4, dropout=0.1, attn_order={1: 0}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0),
    Params(lr=1e-4, dropout=0.15, attn_order={0: 1, 1: 0}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0),
    Params(lr=1e-4, dropout=0.15, attn_order={0: 1}, optim_params={"T_max": 150, "eta_min": 1e-6}, weight_decay=5e-4, img_types=(mods[1], mods[0]), label_smoothing=0.0)
]


def train():
    config = get_mgmt_config()
    with open('run.txt', 'r') as f:
        run = int(f.read())
    with open('run.txt', 'w') as f:
        f.write(str(run + 1))
    for i, params in enumerate(params_list):
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/cross",           
            monitor="val_auc_roc",          
            filename="{epoch:02d}-{val_auc_roc:.4f}" + f'_{run}_{i}', 
            save_top_k=5,                   
            mode="max",                      
        )

        # fixed_epoch_checkpoint = ModelCheckpoint(
        #     dirpath=f"{file_path}/checkpoints/cross",
        #     filename="{epoch:02d}" + f'_{run}_{i}',
        #     every_n_epochs=50,  # Save every 50 epochs
        #     save_top_k=-1,  # Save all models at these epochs
        # )
        logger = TensorBoardLogger(save_dir=f"lightning_logs/cross", name=f"{run}_{i}")

        config = modify_config(config, params)
        config = modify_config(config, {'num_modalities': len(params.img_types)})
        model = Model(config)
        



        data = pd.read_csv("labels.csv")
 
        data = clean_data(data, config.target)
    

        train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=3504)
        sampler = create_sampler(train_df)

        

        val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=3504)


        train_dataset = BrainDataset(config=config, data=train_df, is_train=True, types=params.img_types)
    
        val_dataset = BrainDataset(config=config, data=val_df, is_train=False, types=params.img_types)

        test_dataset = BrainDataset(config=config, data=val_df, is_train=False, types=params.img_types)




        train_loader = DataLoader(train_dataset, batch_size=14, num_workers=5, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=14, shuffle=False, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False, num_workers=5)

        torch.cuda.empty_cache()
        trainer = L.Trainer(
        max_epochs=250,
        accelerator="auto",
        logger=logger,
        devices=2,
        num_nodes=1,
        callbacks=[checkpoint_callback]
        )
        trainer.fit(model, train_loader, val_loader)


def test(params):
    config = get_mgmt_config()
    for file in os.listdir('checkpoints'):
        if 'epoch' not in file:
            continue 



        config = modify_config(config, params)
        config = modify_config(config, {'num_modalities': len(params.img_types)})

        model = Model.load_from_checkpoint(f'checkpoints/{file}', config=config)



        data = pd.read_csv("labels.csv")
        data = clean_data(data, config.target)


        train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=3504)


        

        val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=3504)






        test_dataset = BrainDataset(config=config, data=val_df, is_train=False, types=params.img_types)
     

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=5)

        torch.cuda.empty_cache()
        trainer = L.Trainer(
        max_epochs=250,
        accelerator="auto"
        )
        # trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)
    

train()

