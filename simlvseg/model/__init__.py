import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

import monai

from .unet_3d import UNet3D, UNet3DSmall
from .uniformer import UniFormerUNet
from .vit import ViTUNET

def get_model(
    encoder_name,
    weights=None,
    pretrained_type='encoder',
    img_size=None,
):
    if ('resnet' in encoder_name.lower()) \
        or ('efficientnet' in encoder_name.lower()) \
        or ('mobilenet' in encoder_name.lower()):
        if (pretrained_type.lower() == 'encoder') or (weights is None):
            model = smp.Unet(
                encoder_name,
                encoder_weights = weights,
                in_channels = 3,
                classes = 1,
                activation = None,
            )
        elif pretrained_type.lower() == 'full':
            model = smp.Unet(
                encoder_name,
                encoder_weights = None,
                in_channels = 3,
                classes = 1,
                activation = None,
            )
            # TODO: Implement loading wieghts for full model
            raise NotImplementedError("Not implemented ...")
    elif encoder_name.lower() == 'uniformer_small':
        model = UniFormerUNet(
            "uniformer_small",
            encoder_weights = None,
            in_channels = 3,
            classes = 1,
            activation = None,
        )
        if (weights is not None) and (pretrained_type.lower() == 'encoder'):
            print(f"Loading a pretrained encoder from {weights}")
            state_dict = torch.load(weights, map_location='cpu')
            model.encoder.load_state_dict(state_dict['model'])
        elif (weights is not None) and (pretrained_type.lower() == 'pl_full'):
            print(f"Loading a pretrained encoder-decoder from {weights} (a pytorch_lightning checkpoint)")
            state_dict = torch.load(weights, map_location='cpu')
            
            _temp = TempModule(model)
            _temp.load_state_dict(state_dict['state_dict'])
            
            model = _temp.get_model()
    elif encoder_name.lower() == 'vit_384':
        if pretrained_type.lower() == 'encoder':
            if weights == 'imagenet':
                model = ViTUNET('vit_base_patch16_384', img_size, pretrained=True)
            elif weights is None:
                model = ViTUNET('vit_base_patch16_384', img_size, pretrained=False)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif encoder_name.lower() == 'vit_224':
        if pretrained_type.lower() == 'encoder':
            if weights == 'imagenet':
                model = ViTUNET('vit_base_patch16_224', img_size, pretrained=True)
            elif weights is None:
                model = ViTUNET('vit_base_patch16_224', img_size, pretrained=False)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    # elif encoder_name.lower() == 'monai_2d_unet':
    #     model = monai.networks.nets.UNet(
    #         spatial_dims=2,
    #         in_channels=3,
    #         out_channels=1,
    #         channels=(16, 32, 64, 128, 256, 512),
    #         strides=(2, 2, 2, 2, 2),
    #         num_res_units=8,
    #     )
    elif encoder_name.lower() == 'monai_3d_unet':
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 256),
            strides=(2, 2, 2, 2, 1),
            num_res_units=6,
        )
    elif encoder_name.lower() == 'monai_3d_unet_v2':
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            channels=(16, 16, 16,
                      32, 32, 32, 32,
                      64, 64, 64, 64,
                      128, 128, 128,
                      256, 256, 256,
                    ),
            strides=(1, 1, 2,
                     1, 1, 1, 2,
                     1, 1, 1, 2,
                     1, 1, 2,
                     1, 1,
                    ),
            num_res_units=2,
        )
    elif encoder_name.lower() == '3d_unet':
        model = UNet3D()
        if (weights is not None) and (pretrained_type.lower() == 'pl_full'):
            print(f"Loading a pretrained encoder-decoder from {weights} (a pytorch_lightning checkpoint)")
            state_dict = torch.load(weights, map_location='cpu')
            
            _temp = TempModule(model)
            _temp.load_state_dict(state_dict['state_dict'])
            
            model = _temp.get_model()
    elif encoder_name.lower() == '3d_unet_small':
        model = UNet3DSmall()
        if (weights is not None) and (pretrained_type.lower() == 'pl_full'):
            print(f"Loading a pretrained encoder-decoder from {weights} (a pytorch_lightning checkpoint)")
            state_dict = torch.load(weights, map_location='cpu')
            
            _temp = TempModule(model)
            _temp.load_state_dict(state_dict['state_dict'])
            
            model = _temp.get_model()
    else:
        raise NotImplementedError(f"{encoder_name} is not recognized ...")
    
    return model

class TempModule(pl.LightningModule):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        
        self.model = model
    
    def get_model(self):
        return self.model