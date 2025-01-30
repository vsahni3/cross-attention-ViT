import ml_collections

def get_mgmt_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (4, 4, 4)})
    config.patches.grid = (4, 4, 4)
    config.hidden_dim = 8
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 2
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.patch_size = 4

    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 64)
    config.down_factor = 2
    config.down_num = 2
    config.decoder_channels = (96, 48, 32, 32, 16)
    config.skip_channels = (32, 32, 32, 32, 16)
    config.n_dims = 3
    config.n_skip = 5

    config.img_size = (8, 8, 8)
    config.num_modalities = 2
    config.in_channels = 1
    config.spacing = (2, 2, 2)

    config.target = "MGMT status"
    return config
