import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import lightning as L
import ml_collections
import torchmetrics
from utils import compute_metrics
from torchvision.ops import StochasticDepth
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, config, fn):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, config, dim_head):
        super().__init__()
        inner_dim = dim_head * config.num_heads
        project_out = not (config.num_heads == 1 and dim_head == config.hidden_dim)

        self.heads = config.num_heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(config.hidden_dim, inner_dim * 3, bias = False)
        assert inner_dim == config.hidden_dim
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([])
            
        drop_path_rates = torch.linspace(0, config.drop_path, config.num_layers).tolist()
        
        # if we use same droppath for both they share whether being dropped or not at same time
        for drop_path_rate in drop_path_rates:
            self.layers.append(nn.ModuleList([
                PreNorm(config, Attention(config, dim_head=(config.hidden_dim // config.num_heads))),
                StochasticDepth(drop_path_rate, mode="row"),  # DropPath for MHA
                PreNorm(config, FeedForward(config)),
                StochasticDepth(drop_path_rate, mode="row"),  # DropPath for FFN
            ]))
    def forward(self, x):
        for attn, drop_path_attn, ff, drop_path_ff in self.layers:
            x = drop_path_attn(attn(x)) + x
            x = drop_path_ff(ff(x)) + x
        return x

class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        assert all(config.img_size[i] % config.patch_size[i] == 0 for i in range(len(config.img_size))), 'image dimensions must be divisible by the patch size'
        D, H, W = config.img_size
        dp, hp, wp = config.patch_size
        num_patches = (D // dp) * (H // hp) * (W // wp) * config.num_modalities

        patch_dim = dp * hp * wp

        self.patch_size = config.patch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.optim_params = config.optim_params

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)

        self.transformer = Transformer(config)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.num_classes),
            nn.Dropout(config.dropout)
        )
    def forward(self, img, labels):

        dp, hp, wp = self.patch_size
        all_tokens = []
        for modality in range(img.shape[1]):
            
            cur_x = rearrange(img.select(1, modality), 'b c (d p1) (h p2) (w p3) -> b (h w d) (p1 p2 p3 c)', p1 = dp, p2 = hp, p3 = wp)
            cur_x = self.patch_to_embedding(cur_x)
          
            all_tokens.append(cur_x)
        x = torch.cat(all_tokens, dim=1)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        loss = F.cross_entropy(x, labels)
        return x, loss
    
    def log_stats(self, name, logits, labels):
        pred = torch.argmax(logits, dim=1)
        metrics = compute_metrics(pred, labels)
        self.log(f'{name}_acc', metrics['accuracy'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_prec', metrics['precision'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_rec', metrics['recall'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_spec', metrics['specificity'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_f1', metrics['f1_score'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_npv', metrics['npv'], on_epoch=True, on_step=False, sync_dist=True)

        prob = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        auroc = torchmetrics.functional.auroc(prob, labels, task="binary")
        self.log(f'{name}_auc_roc', auroc, on_epoch=True, on_step=False, sync_dist=True)




    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits, loss = self(x, labels)

        
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log_stats('train', logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits, loss = self(x, labels)

        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log_stats('val', logits, labels)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # note uses half cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optim_params["T_max"],  # Number of epochs before stop decaying
            eta_min=self.optim_params["eta_min"]  # Minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" 
            }
        }
        
    def test_step(self, batch, batch_idx):
        x, labels = batch
        logits, loss = self(x, labels)

        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log_stats('test', logits, labels)
        
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode="min", factor=self.optim_params['factor'], patience=self.optim_params['patience']
    #     )
        
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": self.optim_params['type']
    #         },
    #     }

