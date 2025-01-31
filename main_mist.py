from dataset_ucsf import BrainDataset, clean_data
from modelv2 import ViT3D
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch
from utils import compute_metrics
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data.distributed import DistributedSampler 
import config
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






# class BrainDenseNet(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.loss_fn = torch.nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)

    #     loss = self.loss_fn(logits, y)
    #     print(loss)
    #     self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    #     preds = torch.argmax(logits, dim=1)

    #     metrics = compute_metrics(preds, y)
    #     self.log('train_acc', metrics['accuracy'], on_epoch=True, sync_dist=True)
    #     self.log('train_prec', metrics['precision'], on_epoch=True, sync_dist=True)
    #     self.log('train_rec', metrics['recall'], on_epoch=True, sync_dist=True)
    #     self.log('train_spec', metrics['specificity'], on_epoch=True, sync_dist=True)
    #     self.log('train_f1', metrics['f1_score'], on_epoch=True, sync_dist=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)

    #     loss = self.loss_fn(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    #     metrics = compute_metrics(preds, y)
    #     self.log('val_acc', metrics['accuracy'], on_epoch=True, sync_dist=True)
    #     self.log('val_prec', metrics['precision'], on_epoch=True, sync_dist=True)
    #     self.log('val_rec', metrics['recall'], on_epoch=True, sync_dist=True)
    #     self.log('val_spec', metrics['specificity'], on_epoch=True, sync_dist=True)
    #     self.log('val_f1', metrics['f1_score'], on_epoch=True, sync_dist=True)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_loss",
    #         },
    #     }


def reset_weights(m):
    """
    Resets the weights of a model using appropriate initialization methods.
    """
    if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Reset bias to zero
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.ones_(m.weight)  # Set scaling factor to 1
        nn.init.zeros_(m.bias)  # Reset bias to zero

# Instantiate the model

# model.apply(reset_weights)
Params = namedtuple("Params", ["lr", "drop", "sched_type", "patience", "factor", "weight_decay", "img_types"])


params_list = [
    Params(lr=1e-4, drop=0.0, sched_type='train_loss', patience=25, factor=0.05, weight_decay=1e-3, img_types=("T1c", "T2")),
    Params(lr=1e-4, drop=0.1, sched_type='train_loss', patience=25, factor=0.05, weight_decay=1e-3, img_types=("T1c", "T2")),
    Params(lr=1e-4, drop=0.1, sched_type='train_loss', patience=25, factor=0.0, weight_decay=1e-3, img_types=("T1c", "T2"))
]

legend = 'lr drop val_or_train_sched patience factor weight_decay shape img'
for params in params_list:

    params_title = f'{params.lr} {params.drop} {params.sched_type} {params.patience} {params.factor} {params.weight_decay} {params.img_types}'
    logger = TensorBoardLogger(save_dir=f"{file_path}/lightning_logs", name=f"vit_model_{params_title}")
    # Model
    # model = ViT(config)
    model = ViT3D(
        config=config,
        num_classes=2, 
        add_cls_token=True,
        dropout=params.drop,
        lr=params.lr,
        weight_decay=params.weight_decay,
        optimizer_params={
            'type': params.sched_type,
            'patience': params.patience,
            'factor': params.factor
        }
    )




    data = pd.read_csv("labels.csv")
    data = clean_data(data, config.target)


    train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=6969)

    num_negative = len(train_df[train_df[config.target] == 0])
    num_positive = len(train_df) - num_negative

    # neg must be first cuz of indexing by label
    class_counts = [num_negative, num_positive]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[int(label)] for label in train_df[config.target]]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=6969)

    train_dataset = BrainDataset(data=train_df, is_train=True, types=params.img_types)
    val_dataset = BrainDataset(data=val_df, is_train=False, types=params.img_types)
    # test_dataset = BrainDataset(data=test_df, is_train=False)




    train_loader = DataLoader(train_dataset, batch_size=12, sampler=sampler, num_workers=5)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=5)
    # test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=5)

    torch.cuda.empty_cache()
    trainer = L.Trainer(
    max_epochs=250,
    logger=logger,
    accelerator="gpu",  
    strategy="ddp",      
    devices=4,          
    num_nodes=2,         
    callbacks=[checkpoint_callback]
)
    trainer.fit(model, train_loader, val_loader)





# model = ViT3D(
    #     in_channels=1,
    #     hidden_dim=64,
    #     num_heads=8,
    #     num_layers=12,
    #     num_classes=2, 
    #     img_size=(128, 128, 64),
    #     add_cls_token=True,
    #     dropout=params.drop,
    #     lr=params.lr,
    #     weight_decay=params.weight_decay,
    #     optimizer_params={
    #         'type': params.sched_type,
    #         'patience': params.patience,
    #         'factor': params.factor
    #     }
    # )

