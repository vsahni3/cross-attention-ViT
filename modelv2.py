import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
from utils import compute_metrics
from monai.networks.nets import DenseNet121
from modify_model import get_model_upto_layer
import ml_collections
import torchmetrics
import torch
from torch.profiler import profile, record_function, ProfilerActivity




class CNN3DEncoder(nn.Module):
    """
    A simple 3D CNN that encodes a 3D volume into a smaller feature map.
    """
    def __init__(self, 
                 in_channels: int = 1, 
                 hidden_dim: int = 128):
        """
        Args:
            in_channels: Number of channels in the input volume (e.g., 1 for grayscale, 3 for RGB).
            hidden_dim: The output feature dimension from the CNN.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm3d(128)
        
        self.conv3 = nn.Conv3d(128, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm3d(hidden_dim)
        
        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, in_channels, D, H, W)
        Returns:
            Feature map of shape (B, hidden_dim, D//8, H//8, W//8)
            after three 2x downsampling steps in each dimension.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # downsample by factor of 2

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # downsample by factor of 2

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # downsample by factor of 2

        return x


class TransformerEncoder(nn.Module):
    """
    A simple wrapper around nn.TransformerEncoder to process a sequence of tokens.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=4*embed_dim,  # typical choice is 4 x d_model
            dropout=dropout,
            batch_first=True  # ensures (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, N, embed_dim), where N = number of tokens
        Returns:
            Transformer-encoded features of the same shape (B, N, embed_dim).
        """
        return self.transformer(x)

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


class ViT3D(L.LightningModule):
    def __init__(self, 
                 optimizer_params: dict,
                 lr: float,
                 weight_decay: float,
                 config: ml_collections.ConfigDict,
                 num_classes: int = 2,
                 add_cls_token: bool = True,
                 pretrained_cnn: bool = False,
                 cnn_out_dim: tuple = (64, 8, 8, 8),
                 dropout: float = 0.0,
                 growth_rate: int = 16):
        """
        Args:
            in_channels: Number of channels in the input 3D volume.
            hidden_dim: Dimensionality of tokens/features inside the Transformer.
            num_heads: Number of attention heads in the Transformer encoder.
            num_layers: Number of layers in the Transformer encoder.
            num_classes: Number of output classes for classification tasks.
            dropout: Dropout probability.
            add_cls_token: Whether to prepend a [CLS] token for classification.
        """
        super().__init__()
        self.lr = lr
        self.optimizer_params = optimizer_params
        self.weight_decay = weight_decay
        if pretrained_cnn:
            raw_model = DenseNet121(
                spatial_dims=3,  
                in_channels=1,  
                out_channels=2,
                dropout_prob=dropout,
                growth_rate=growth_rate
            )
            raw_model.apply(reset_weights)
            self.encoder_3d = get_model_upto_layer(raw_model, "features.denseblock3.denselayer24.layers.conv1")
        else:
            self.encoder_3d = CNN3DEncoder(hidden_dim=config.hidden_dim)
        
        # 2. [CLS] token
        self.add_cls_token = add_cls_token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        else:
            self.cls_token = None

        
        # implicitly determined
        
        if pretrained_cnn:
            num_tokens = cnn_out_dim[1] * cnn_out_dim[2] * cnn_out_dim[3]
        else:
            D, H, W = config.img_size
            num_tokens = (D // 8) * (H // 8) * (W // 8) * config.num_modalities
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + int(add_cls_token), config.hidden_dim))
        
        # 3. Transformer encoder
        self.transformer = TransformerEncoder(embed_dim=config.hidden_dim,
                                              num_heads=config.transformer.num_heads,
                                              num_layers=config.transformer.num_layers,
                                              dropout=dropout)
        
        # 4. Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, num_classes),
        )
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels) -> torch.Tensor:
        """
        Args:
            x: shape (B, in_channels, D, H, W)
        Returns:
            logits: shape (B, num_classes)
        """
        # 1. Pass through the 3D CNN encoder
        #    Output shape: (B, hidden_dim, D', H', W')
        all_tokens = []
        for modality in range(x.shape[1]):
            cur_x = self.encoder_3d(x.select(1, modality))
            
            
            B, C, Dp, Hp, Wp = cur_x.shape  # e.g. B, 128, D/8, H/8, W/8
            
            # 2. Flatten spatial dims => sequence of tokens
            #    shape after flatten: (B, C, D'*H'*W')
            cur_x = cur_x.flatten(start_dim=2)  # (B, C, N), N = D'*H'*W'
            all_tokens.append(cur_x)
        x = torch.cat(all_tokens, dim=2)
        
        #    transpose to get (B, N, C)
        x = x.transpose(1, 2)  # (B, N, C)

        N = x.size(1)          # number of tokens

        # 3. Add [CLS] token if we’re using it
        if self.add_cls_token and self.cls_token is not None:
            # Expand the class token across the batch dimension
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
            x = torch.cat((cls_tokens, x), dim=1)          # (B, N+1, C)
        
        # 4. Add positional embeddings
        x = x + self.pos_embed  # broadcast over batch
        
        # 5. Pass through Transformer
        x = self.transformer(x)  # (B, N+1, C) or (B, N, C)
        with open(f'file.txt', 'a') as f:
            f.write(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
            f.write(f"Reserved Memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB\n")
            f.write(f'{torch.cuda.memory_summary()}\n')
        
        # 6. Classification head: use the [CLS] token output
        if self.add_cls_token:
            # x[:, 0] is the [CLS] token representation
            cls_out = x[:, 0]             # (B, C)
        else:
            # Alternatively, some ViTs pool the mean of all tokens, for example
            cls_out = x.mean(dim=1)       # (B, C)
      
        
        logits = self.mlp_head(cls_out)    # (B, num_classes)
        loss = F.cross_entropy(logits, labels)
        return logits, loss

    def log_stats(self, is_train, logits, labels):
        name = 'train' if is_train else 'val'
        pred = torch.argmax(logits, dim=1)
        metrics = compute_metrics(pred, labels)
        self.log(f'{name}_acc', metrics['accuracy'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_prec', metrics['precision'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_rec', metrics['recall'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_spec', metrics['specificity'], on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{name}_f1', metrics['f1_score'], on_epoch=True, on_step=False, sync_dist=True)

        prob = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        auroc = torchmetrics.functional.auroc(prob, labels, task="binary")
        self.log(f'{name}_auc_roc', auroc, on_epoch=True, on_step=False, sync_dist=True)




    def training_step(self, batch, batch_idx):
        x, labels = batch
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                logits, loss = self(x, labels)
        with open('file.txt', 'a') as f:
            f.write(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10) + '\n\n')
        
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log_stats(True, logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits, loss = self(x, labels)

        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log_stats(False, logits, labels)

        



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.optimizer_params['factor'], patience=self.optimizer_params['patience']
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.optimizer_params['type']
            },
        }
