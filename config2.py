import ml_collections

def get_mgmt_config():
    config = ml_collections.ConfigDict()
    config.hidden_dim = 512
    config.mlp_dim = 2048
    config.num_heads = 16
    config.num_layers = 4
    config.patch_size = (16, 16, 16)

    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 64)
    config.down_factor = 2
    config.down_num = 2


    config.num_classes = 2
    config.img_size = (256, 256, 128)
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