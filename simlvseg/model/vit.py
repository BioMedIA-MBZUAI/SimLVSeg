# Reference:
# https://github.com/rwightman/pytorch-image-models/blob/9ecab46e180af997b345427c2ab11ff1b6c0dfb9/timm/models/vision_transformer.py
# https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/networks/unetr.py

import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import VisionTransformer
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from timm.models.vision_transformer import checkpoint_filter_fn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

def create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        ViT, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model

class ViT(VisionTransformer):
    def forward(self, x):
        features = []
        
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError
        else:
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in [2, 5, 8]:
                    features.append(x[:,1:])
        x = self.norm(x)
        features.append(x[:,1:])
        
        return features

class ViTUNET(nn.Module):
    def __init__(
        self,
        vit_type='vit_base_patch16_384',
        img_size=384,
        pretrained=True,
        patch_size=16,
        in_channels=3,
        feature_size=16,
        norm_name="instance",
        res_block=True,
        hidden_size=768,
        conv_block=True,
        out_channels=1,
    ):
        super().__init__()
        
        if vit_type == 'vit_base_patch16_384':
            self.vit = vit_base_patch16_384(pretrained=pretrained, img_size=img_size)
        elif vit_type == 'vit_base_patch16_224':
            self.vit = vit_base_patch16_224(pretrained=pretrained, img_size=img_size)
        else:
            raise NotImplementedError
        
        self.hidden_size = hidden_size
        self.feat_size = img_size // patch_size
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)  # type: ignore
    
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size, feat_size, hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def forward(self, x_in):
        features = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = features[0]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = features[1]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = features[2]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(features[3], self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits