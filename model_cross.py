import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import lightning as L
import ml_collections
import torchmetrics
from utils import compute_metrics

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
    

class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = PreNorm(config, Attention(config, dim_head=(config.hidden_dim // config.num_heads)))
        self.ffn = PreNorm(config, FeedForward(config))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_dim // config.num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.wk = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.wv = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.proj_drop = nn.Dropout(config.dropout)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttentionBlock(nn.Module):

    def __init__(self, config, act_layer=nn.GELU):
        super().__init__()
        self.attn = PreNorm(config, CrossAttention(config))
        self.ffn = PreNorm(config, FeedForward(config))

    def forward(self, x):
        x = self.attn(x) + x[:, 0:1, ...]
        x = self.ffn(x) + x
        return x
    
class MultiScaleBlock(nn.Module):

    def __init__(self, config, act_layer=nn.GELU):
        super().__init__()
        self.attn_order = config.attn_order
        # separate set of blocks as each modality has diff features to learn
        self.blocks = nn.ModuleList([nn.Sequential(*[SelfAttentionBlock(config) for _ in range(config.num_self_blocks)]) for _ in range(config.num_modalities)])
        # note: no need for projection and unprojection cls token as dims of all embeddings are same for banches
        self.fusion = nn.ModuleList([CrossAttentionBlock(config) for i in range(len(self.attn_order))])
        


    def forward(self, x):

        attn = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        # cross attention
        outs = []
        for i in range(len(self.blocks)):
            if i in self.attn_order:
                idx_cls = i
                idx_tokens = self.attn_order[idx_cls]
                # not using select cuz need to keep outer dim for cat
                tmp = torch.cat((attn[idx_cls][:, 0:1, ...], attn[idx_tokens][:, 1:, ...]), dim=1)
                tmp = self.fusion[i](tmp)
                tmp = torch.cat((tmp, attn[idx_cls][:, 1:, ...]), dim=1)
                outs.append(tmp)
            else:
                outs.append(attn[i])
        
        return outs
    


class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        assert all(config.img_size[i] % config.patch_size[i] == 0 for i in range(len(config.img_size))), 'image dimensions must be divisible by the patch size'
        D, H, W = config.img_size
        dp, hp, wp = config.patch_size
        num_patches = (D // dp) * (H // hp) * (W // wp)

        patch_dim = dp * hp * wp

        self.patch_size = config.patch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.optim_params = config.optim_params
        self.num_modalities = config.num_modalities
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)

        self.transformer = nn.Sequential(*[MultiScaleBlock(config) for _ in range(config.num_multi_blocks)])
        # different params for each branch, ffn for block is already specific to block so ok, but here we are doing for all blocks
        self.norm = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.num_modalities)])
        
        self.mlp_head = nn.ModuleList([
            nn.Sequential(
            nn.Linear(config.hidden_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.num_classes),
            nn.Dropout(config.dropout)
        ) for _ in range(config.num_modalities)])
        self.initialize_model()
    def forward(self, img, labels):
        

        dp, hp, wp = self.patch_size
        all_tokens = []
        for modality in range(img.shape[1]):
            
            x = rearrange(img.select(1, modality), 'b c (d p1) (h p2) (w p3) -> b (h w d) (p1 p2 p3 c)', p1=dp, p2=hp, p3=wp)
            x = self.patch_to_embedding(x)
            cls_token = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x += self.pos_embedding
            x = self.dropout(x)
          
            all_tokens.append(x)

        x = self.transformer(all_tokens)
        x = [self.norm[i](x[i]) for i in range(len(x))]
        x = torch.stack([self.mlp_head[i](x[i][:, 0]) for i in range(self.num_modalities)])
        x = torch.mean(x, dim=0)
        loss = F.cross_entropy(x, labels)
        return x, loss
    
    @staticmethod
    def init_weights(module):
        """
        Custom weight initialization for Linear and LayerNorm layers.
        """
        if isinstance(module, nn.Linear):
            # Use Xavier Uniform initialization for Linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm layers: weight=1, bias=0
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def initialize_model(self):
        """
        Apply weight initialization to all submodules and reinitialize
        the positional embeddings and class token.
        """
        # Initialize modules (Linear, LayerNorm, etc.)
        self.apply(Model.init_weights)
        
        # Reinitialize parameters that were defined as nn.Parameter
        if hasattr(self, 'pos_embedding'):
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
    
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
        
