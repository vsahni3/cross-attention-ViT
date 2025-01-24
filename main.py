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
from monai.networks.nets import DenseNet121
from sklearn.model_selection import train_test_split
import torch.nn as nn
from modify_model import get_model_upto_layer




# DIM / IMAGE SIZE ONLY 128 MAYBE UPSCALE IF NEEDED
# Load configuration
config = config.get_mgmt_config()

# Callbacks
checkpoint_callback = ModelCheckpoint(
   dirpath="checkpoints",
   monitor="val_loss",
   filename="vit-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
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

logger = TensorBoardLogger(save_dir="lightning_logs", name="vit_model")
# Model
# model = ViT(config)
model = ViT3D(
    in_channels=1,
    hidden_dim=64,
    num_heads=8,
    num_layers=12,
    num_classes=2, 
    img_size=(128, 128, 128),
    add_cls_token=False,
    dropout=0.0
)

# Dataset
data = pd.read_csv("labels.csv")
data = clean_data(data, config.target)


train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=6969)
# train_df = train_df.iloc[:3]
num_negative = len(train_df[train_df[config.target] == 0])
num_positive = len(train_df) - num_negative

# neg must be first cuz of indexing by label
class_counts = [num_negative, num_positive]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[int(label)] for label in train_df[config.target]]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=6969)
# val_df = val_df.iloc[:3]

train_dataset = BrainDataset(data=train_df, is_train=True)
val_dataset = BrainDataset(data=val_df, is_train=False)
test_dataset = BrainDataset(data=test_df, is_train=False)




train_loader = DataLoader(train_dataset, batch_size=5, num_workers=5, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=5)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=5)


trainer = L.Trainer(
max_epochs=100,
logger=logger,
accelerator="auto",
devices=2,
callbacks=[checkpoint_callback]
)
trainer.fit(model, train_loader, val_loader)
