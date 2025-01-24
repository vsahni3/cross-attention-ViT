import copy
import logging
import math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import pandas as pd
import config
config = config.get_mgmt_config()
from torch.distributions.normal import Normal
from dataset_ucsf import BrainDataset
from time import time 
import lightning as L
from utils import compute_metrics
from functools import reduce 

# works with just cnn encoder and 2 linear layers
# works with above + patch embed
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x) # due to padding dim not changed, only channels changed


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x) # each dim is halved

class CNNEncoder(nn.Module):
    # goal of this is to just extract features to make it easier for vit, we shouldnt fuse modalities through the cnn
    # so we cant treat each modality as a channel as then it'll get prematurely fused
    def __init__(self, config, n_channels=1):
        # (B, 1, image_size, image_size, num_images)
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.inc = DoubleConv(n_channels, encoder_channels[0])  # (B, encoder_channels[0], image_size, image_size, num_images)
        self.down1 = Down(encoder_channels[0], encoder_channels[1]) # (B, encoder_channels[1], image_size / 2, image_size / 2, num_images / 2)
        self.down2 = Down(encoder_channels[1], encoder_channels[2]) # (B, encoder_channels[2], image_size / 4, image_size / 4, num_images / 4)
 

    def forward(self, x):
        x = self.inc(x)

        x = self.down1(x)

        x = self.down2(x)

        return x
    


class Embeddings(nn.Module):
    def __init__(self, config, n_channels=1):
        # experiment with dim reduction of num_images and patch size
        super(Embeddings, self).__init__()
        self.cnn_encoder = CNNEncoder(config, n_channels) # (B, encoder_channels[2], image_size / 4, image_size / 4, num_images / 4)
        self.patch_embed = nn.Conv3d(config.encoder_channels[2], config.hidden_size, kernel_size=config.patches.grid, stride=config.patches.grid) # (B, hidden_size, (image_size / 4) / patches.grid[0], (image_size / 4) / patches.grid[1], (num_images / 4) / patches.grid[2])
        # look at above dim comment for patch embed for how this is calculated
        num_patches = (config.img_size[0] / (2 ** config.down_factor * config.patches.grid[0])) * (config.img_size[1] / (2 ** config.down_factor * config.patches.grid[1])) * (config.img_size[2] / (2 ** config.down_factor * config.patches.grid[2])) 

        self.class_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, int(num_patches + 1), config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
    def forward(self, x):
        # note for each modality self.positonal_embedding is the same 
        x = self.cnn_encoder(x)
        x = self.patch_embed(x)


        x = x.flatten(-3)
        x = x.transpose(-2, -1)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.concat((class_token, x), dim=1)
        x = x + self.positional_embedding
        # x = self.dropout(x)

        return x


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
        
class MultiHeadAttention(nn.Module):
    # vectorized instead of looping through heads

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
    
    def split_into_heads(self, x):
        # can't just call reshape directly as then it will not be correctly split into heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        concat_shape = x.shape[:-2] + (self.all_head_size,)
        x = x.reshape(concat_shape)
        return x
    

    def forward(self, x):
        # x is (B, num_tokens, embedding_dim)

        # (B, n_heads, seq_size, head_size)
        query = self.split_into_heads(self.query(x))
        key = self.split_into_heads(self.key(x))
        value = self.split_into_heads(self.value(x))

        # (B, n_heads, num_tokens, num_tokens)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # (B, n_heads, num_tokens, head_size)
        contextualized_layer = torch.matmul(attention_probs, value)

        contextualized_layer = self.merge_heads(contextualized_layer)

        attention_output = self.out(contextualized_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.multi_head = MultiHeadAttention(config)
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)

    def forward(self, x):
        # pre norm - residual unnoramilzied straightforward gradient, inputs normalized before going into layer
        old_x = x
        x = self.attention_norm(x)
        x = self.multi_head(x)
        x = old_x + x 

        old_x = x 
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = old_x + x
        return x

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.layers = nn.Sequential(*[copy.deepcopy(Block(config)) for _ in range(config.transformer["num_layers"])])

    
    def forward(self, x):
        x = self.layers(x)
        x = self.encoder_norm(x)
        return x


class ViT(L.LightningModule):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        # (B, seq_size, embed_dim)
        self.final = Linear(128, 1)
        # self.a = Linear(32, 1)

        self._init_weights()
        # self.model = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # Conv layer
        #     nn.ReLU(),  # Activation
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # Downsampling
        #     nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Another Conv layer
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),
        #     nn.Flatten(),  # Flatten before dense layers
        #     nn.Linear(32 * 32 * 32 * 32, 128),  # Fully connected layer
        #     nn.ReLU(),
        #     nn.Linear(128, 1)  # Output layer
        # )
        self.loss = torch.nn.BCEWithLogitsLoss()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, label=None):
        
        # (B, num_modalities, 1, image_size, image_size, num_images)
        # print(torch.sum(torch.abs(x[0] - x[1])), 'FIRST', torch.sum(torch.abs(x[0] - x[1])) / reduce(lambda x, y: x * y, x.shape[1:]))
        # only use cls toke from first
        x = torch.cat([self.embeddings(x.select(1, 0))] + [self.embeddings(x.select(1, i))[:, 1:, :] for i in range(1, x.shape[1])], dim=1)
        # x = x.squeeze(1)
        
        # x = x.flatten(1)
   



        # print(torch.sum(torch.abs(x[0] - x[1])), 'SECOND', torch.sum(torch.abs(x[0] - x[1])) / reduce(lambda x, y: x * y, x.shape[1:]))
        x = self.encoder(x)
        # print(torch.sum(torch.abs(x[0] - x[1])), 'THIRD', torch.sum(torch.abs(x[0] - x[1])) / reduce(lambda x, y: x * y, x.shape[1:]))
        # cls token
        x = x[:, 0, :]
        # print(torch.sum(torch.abs(x[0] - x[1])), 'FOURTH', torch.sum(torch.abs(x[0] - x[1])) / reduce(lambda x, y: x * y, x.shape[1:]))
    
        
        # x = self.a(torch.relu(self.final(x))).squeeze(-1)
        x = self.final(x).squeeze(-1)
        # x = self.model(x).squeeze(-1)

        # print(x, label, 'AHHHHH')


        if label is None:
            return x 

        

        return x, self.loss(x, label)




    def training_step(self, batch, batch_idx):
        x, labels = batch

        prob, loss = self(x, labels)
        print(loss)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        metrics = compute_metrics(prob, labels)
        self.log('train_acc', metrics['accuracy'], on_epoch=True, sync_dist=True)
        self.log('train_prec', metrics['precision'], on_epoch=True, sync_dist=True)
        self.log('train_rec', metrics['recall'], on_epoch=True, sync_dist=True)
        self.log('train_spec', metrics['specificity'], on_epoch=True, sync_dist=True)
        self.log('train_f1', metrics['f1_score'], on_epoch=True, sync_dist=True)

    
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        prob, loss = self(x, labels)

        self.log('val_loss', loss, on_epoch=True, sync_dist=True)

        # metrics = compute_metrics(prob, labels)
        # self.log('val_acc', metrics['accuracy'], on_epoch=True, sync_dist=True)
        # self.log('val_prec', metrics['precision'], on_epoch=True, sync_dist=True)
        # self.log('val_rec', metrics['recall'], on_epoch=True, sync_dist=True)
        # self.log('val_spec', metrics['specificity'], on_epoch=True, sync_dist=True)
        # self.log('val_f1', metrics['f1_score'], on_epoch=True, sync_dist=True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }



data = pd.read_csv("labels.csv")
train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=6969)
val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=6969)

train_dataset = BrainDataset(data=train_df, is_train=True)
val_dataset = BrainDataset(data=val_df, is_train=False)
test_dataset = BrainDataset(data=test_df, is_train=False)


# from time import time 
# start = time()

# image1 = train_dataset[0][0]
# image2 = train_dataset[1][0]
# print(torch.sum(torch.abs(image1 - image2)))
# # print(image.shape)
# image = torch.stack((image1, image2))
# print(image.shape)
# a = ViT(config)
# print(a(image, torch.tensor([1.0])))
# print(time() - start)
# tensor = torch.randn(1, 5, 6)
# print(tensor, tensor.view(1, 5, 2, 3), tensor.view(1, 5, 2, 3).contiguous().reshape(1, 2, 5, 3), sep='\n')
# print(tensor.reshape(1, 2, 5, 3))
# print(time() - start)