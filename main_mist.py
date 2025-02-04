from dataset_ucsf import BrainDataset, clean_data
from modelv3 import Model
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data.distributed import DistributedSampler 
import config
from config import modify_config
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from collections import namedtuple




# DIM / IMAGE SIZE ONLY 128 MAYBE UPSCALE IF NEEDED
# Load configuration
config = config.get_mgmt_config()


file_path = '/scratch/p/ptyrrell/vsahni3'
# Callbacks
checkpoint_callback = ModelCheckpoint(
   dirpath=f"{file_path}/checkpoints",
   monitor="train_loss",
   filename="vit-{epoch:02d}",
   save_top_k=3,
   mode="min",
)





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
Params = namedtuple("Params", ["lr", "dropout", "optim_params", "weight_decay", "img_types", "label_smoothing"])
# label smoothing
# stochatsic depth
# precision affects
# shuffle labels to see if truly memorizng
#AJWIDNWEFNIEFNEOJFKEFMEMFE
mods = ['DWI', 'SWI', 'T1c', 'brain_parenchyma_segmentation', 'tumor_segmentation', 'T2', 'ADC', 'ASL']
params_list = [
    Params(lr=1e-4, dropout=0.1, optim_params={}, weight_decay=1e-3, img_types=(mods[1], mods[2]), label_smoothing=0.0),
    Params(lr=1e-4, dropout=0.1, optim_params={}, weight_decay=1e-3, img_types=(mods[0], mods[1]), label_smoothing=0.0),
    Params(lr=1e-4, dropout=0.1, optim_params={}, weight_decay=1e-3, img_types=(mods[3], mods[1]), label_smoothing=0.0),
]


for params in params_list:
    params_title = f'{params.lr} {params.dropout} {params.optim_params} {params.weight_decay} {params.img_types} {params.label_smoothing}'
    logger = TensorBoardLogger(save_dir=f"{file_path}/lightning_logs", name=f"vit_model_{params_title}")

    config = modify_config(config, params)
    config = modify_config(config, {'num_modalities': len(params.img_types)})
    model = Model(config)



    data = pd.read_csv("labels.csv")
    data = clean_data(data, config.target)


    train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=6969)
    sampler = create_sampler(train_df)

    

    val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=6969)


    train_dataset = BrainDataset(data=train_df, is_train=True, types=params.img_types)
    val_dataset = BrainDataset(data=val_df, is_train=False, types=params.img_types)


    # test_dataset = BrainDataset(data=test_df, is_train=False)




    train_loader = DataLoader(train_dataset, batch_size=12, num_workers=5, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=5)
    # test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=5)

    torch.cuda.empty_cache()
    trainer = L.Trainer(
    max_epochs=250,
    accelerator="auto",
    logger=logger,
    devices=4,
    num_nodes=2,
    callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)





