import ml_collections

def get_mgmt_config():
    config = ml_collections.ConfigDict()
    config.hidden_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    # for vanilla vit
    config.num_layers = 4
    config.num_multi_blocks = 3
    config.num_self_blocks = 2
    config.patch_size = (16, 16, 8)


    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 64)
    config.down_factor = 2
    config.down_num = 2


    config.num_classes = 2
    config.img_size = (128, 128, 64)
    config.in_channels = 1
    config.spacing = (2, 2, 2)

    config.target = "MGMT status"
    
    return config

def modify_config(config, params):
    if not isinstance(params, dict):
        params = params._asdict()
    for key, value in params.items():
        setattr(config, key, value)
    
    return config