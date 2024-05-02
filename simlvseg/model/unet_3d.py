import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128])
        self.encoder5 = self._create_encoder_block(128, [256, 256, 256])
        
        self.decoder4 = self._create_decoder_block(128*2, 128, False)
        self.decoder3 = self._create_decoder_block(64*2, 64, False)
        self.decoder2 = self._create_decoder_block(32*2, 32, False)
        self.decoder1 = self._create_decoder_block(16*2, 1, True)
        
        self.upconv5 = self._create_up_conv(256, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)
        
        self.maxpool = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        _, _, h, w, d = x.shape
        
        if (h%16 != 0) or (w%16 != 0) or (d%16 != 0):
            raise ValueError(f"Invalid volume size ({h}, {w}, {d}). The dimension need to be divisible by 16.")
        
        x1 = self.encoder1(x)
        x  = self.maxpool(x1)
        
        x2 = self.encoder2(x)
        x  = self.maxpool(x2)
        
        x3 = self.encoder3(x)
        x  = self.maxpool(x3)
        
        x4 = self.encoder4(x)
        x  = self.maxpool(x4)
        
        x = self.encoder5(x)
        
        x = self.upconv5(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        
        return x
    
    def _create_encoder_block(
        self,
        in_channel,
        channels,
        down_sampling=False,
        ):
        
        def _create_residual_unit(
            in_channels, out_channels, strides,
            ):
            return ResidualUnit(
                3,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                subunits=2,
                act=Act.PRELU,
                norm=Norm.INSTANCE,
                dropout=0.0,
                bias=True,
                adn_ordering="NDA",
            )
        
        _channels = [in_channel, *channels]
        
        units = []
        for i in range(len(channels) - 1):
            units.append(_create_residual_unit(_channels[i], _channels[i+1], 1))
        units.append(
            _create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1)
        )
        
        return nn.Sequential(*units)
    
    def _create_decoder_block(
        self,
        in_channels,
        out_channels,
        is_top,
    ):
        res_unit = ResidualUnit(
            3,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=is_top,
            adn_ordering="NDA",
        )
        
        return res_unit
    
    def _create_up_conv(
        self,
        in_channels,
        out_channels,
    ):
        return Convolution(
            3,
            in_channels,
            out_channels,
            strides=2,
            kernel_size=3,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            is_transposed=True,
            adn_ordering="NDA",
        )

class UNet3DSmall(UNet3D):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128])
        self.encoder5 = self._create_encoder_block(128, [256, 256])
        
        self.decoder4 = self._create_decoder_block(128*2, 128, False)
        self.decoder3 = self._create_decoder_block(64*2, 64, False)
        self.decoder2 = self._create_decoder_block(32*2, 32, False)
        self.decoder1 = self._create_decoder_block(16*2, 1, True)
        
        self.upconv5 = self._create_up_conv(256, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)
        
        self.maxpool = nn.MaxPool3d(2, 2)