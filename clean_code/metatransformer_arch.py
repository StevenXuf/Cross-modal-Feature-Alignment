import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

def metatransformer(n_dim_features=768):
    model=nn.Sequential(*[
                    Block(
                    dim=n_dim_features,
                    num_heads=12,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU
                    )
                    for i in range(12)])
    ckpt=torch.load("/data/data_fxu/Meta-Transformer-checkpoint/Meta-Transformer_base_patch16_encoder.pth")
    model.load_state_dict(ckpt,strict=True)
    return model

if __name__=='__main__':
    print(metatransformer())
