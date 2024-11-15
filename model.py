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
config = config.get_3DReg_config()
from torch.distributions.normal import Normal
from dataset import BrainRSNADataset
from time import time 


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
        # (B, 1, image_size, image_size, image_size)
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
        num_patches = (config.image_size / (2 ** config.down_factor * config.patches.grid[0])) * (config.image_size / (2 ** config.down_factor * config.patches.grid[1])) * (config.num_images / (2 ** config.down_factor * config.patches.grid[2])) 

        self.class_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, int(num_patches + 1), config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
    def forward(self, x):
        # note for each modality self.positonal_embedding is the same 
        x = self.cnn_encoder(x)
        x = self.patch_embed(x)

        x = x.flatten(-3)
        x = x.transpose(-2, -1)

        x = torch.concat((self.class_token, x), dim=1)
        x = x + self.positional_embedding
        x = self.dropout(x)

        return x


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # self._init_weights()
    # custom weight init
    # def _init_weights(self):
    #     nn.init.xavier_uniform_(self.fc1.weight)
    #     nn.init.xavier_uniform_(self.fc2.weight)
    #     nn.init.normal_(self.fc1.bias, std=1e-6)
    #     nn.init.normal_(self.fc2.bias, std=1e-6)

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
        # pre norm - residual unnroamilzied straightforward gradient, inputs normalized before going into layer
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


class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        # (B, seq_size, embed_dim)
        self.final = Linear(config.hidden_size, 1)
        self.act_fn = torch.nn.functional.sigmoid 
        self.loss = nn.BCELoss
    def forward(self, x, label=None):
        # (B, 1, image_size, image_size, image_size)
 
        cur = time()
        # only use cls toke from first
        x = torch.cat([self.embeddings(x.select(1, 0))] + [self.embeddings(x.select(1, i))[:, 1:, :] for i in range(1, x.shape[1])], dim=1)
 
        print(time() - cur)
        x = self.encoder(x)
        # cls token
        x = x[:, 0, :]
        x = self.act_fn(self.final(x))
        if label is None:
            return x 
        return x, self.loss(x, label)









start = time()


data = pd.read_csv("train_labels.csv")
train_df, val_df = train_test_split(data, test_size=0.3, random_state=6969)

train_dataset = BrainRSNADataset(data=train_df, ds_type=f"train")



image = train_dataset[0]['image'].unsqueeze(0)
# image = image.permute(1, 0, 2, 3, 4).view(1, 1024, 256, 64).unsqueeze(0)
# image = image[0].unsqueeze(0)
print(image.shape)
# image = image.permute(1, 0, 2, 3, 4).reshape(1, 1024, 256, 64)




a = ViT(config)
print(a(image))
# tensor = torch.randn(1, 5, 6)
# print(tensor, tensor.view(1, 5, 2, 3), tensor.view(1, 5, 2, 3).contiguous().reshape(1, 2, 5, 3), sep='\n')
# print(tensor.reshape(1, 2, 5, 3))
# print(time() - start)