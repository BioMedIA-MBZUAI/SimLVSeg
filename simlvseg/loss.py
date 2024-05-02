import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(nn.Module):
    def __init__(
        self,
        loss_type = 'dice'
    ):
        super().__init__()
        
        if not isinstance(loss_type, list):
            loss_type = [loss_type]
        
        for l in loss_type:
            if l not in ['bce', 'dice', 'mse']:
                raise ValueError(f'Loss type {l} is not recognized ...')
        
        self.bce_loss  = smp.losses.SoftBCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.mse_loss  = nn.MSELoss()
        self.loss_type = loss_type
    
    def forward(
        self,
        preds,
        labels,
    ):
        loss = 0.
        
        if 'bce' in self.loss_type:
            loss += self.bce_loss(preds, labels)
        
        if 'dice' in self.loss_type:
            loss += self.dice_loss(preds, labels)
        
        if 'mse' in self.loss_type:
            loss += self.mse_loss(preds, labels)
        
        return loss