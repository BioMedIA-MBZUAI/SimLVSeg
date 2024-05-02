import torch
import torch.nn as nn

from ..loss import SegLoss
from ..model import get_model
from ..pl_module import BaseModule

class SISSLMaskingModule(BaseModule):
    def __init__(
        self,
        encoder_name,
        weights=None,
        loss_type='mse',
    ):
        super().__init__()
        
        self.model = get_model(encoder_name, weights, 'encoder')
        
        self.criterion = SegLoss(loss_type)
        self.metrics   = {
            'mse': nn.MSELoss(),
        }
    
    def preprocess_batch_imgs(self, imgs):
        return imgs
    
    def postprocess_batch_preds_and_targets(self, preds, targets):
        labels = targets['si'][:, 0:1, :, :]
        return preds, labels
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=3e-4,
            weight_decay=1e-3, amsgrad=True,
            betas=(0.9, 0.95),
        )
        
        return optimizer