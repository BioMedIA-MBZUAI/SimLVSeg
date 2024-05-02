import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import torch

from .model import get_model
from .loss import SegLoss
from .utils import get_crop_from_coors

class BaseModule(pl.LightningModule):
    def preprocess_batch_imgs(self, imgs):
        raise NotImplementedError
    
    def postprocess_batch_preds_and_targets(self, preds, targets):
        raise NotImplementedError
    
    def configure_optimizers(self):
        raise NotImplementedError
    
    def calculate_metrics(self, set_name, preds, labels):
        # Calculate the metrics
        metrics = [[name, fn(preds, labels)] for name, fn in self.metrics.items()]
        
        # Print the metrics on the terminal
        for name, value in metrics:
            self.log(f"{set_name}_{name}", value, prog_bar=True, logger=True)
    
    def val_test_epoch_end(self, set_name, step_outputs):
        preds  = []
        labels = []
        
        for output in step_outputs:
            preds.append(output['batch_preds'])
            labels.append(output['batch_labels'])
        
        preds  = torch.cat(preds)
        labels = torch.cat(labels)
        
        loss = self.criterion(preds, labels)
        
        self.log(f"{set_name}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.calculate_metrics(set_name, preds, labels)
    
    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        
        imgs = self.preprocess_batch_imgs(imgs)
        
        preds = self.forward(imgs)
        
        preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        
        loss = self.criterion(preds, labels)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.calculate_metrics('train', preds, labels)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        
        imgs = self.preprocess_batch_imgs(imgs)
        
        preds = self.forward(imgs)
        
        preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        
        return {'batch_preds': preds, 'batch_labels': labels}
    
    def validation_epoch_end(self, validation_step_outputs):
        return self.val_test_epoch_end('val', validation_step_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, test_step_outputs):
        return self.val_test_epoch_end('test', test_step_outputs)

class SegModule(BaseModule):
    def __init__(
        self,
        encoder_name,
        weights=None,
        pretrained_type='encoder',
        img_size=None,
        loss_type='dice',
    ):
        super().__init__()
        
        self.model = get_model(encoder_name, weights, pretrained_type, img_size)
        
        self.criterion = SegLoss(loss_type)
        self.metrics   = {
            'dsc': smp_utils.metrics.Fscore(activation='sigmoid'),
            'dice_loss': smp.losses.DiceLoss(mode='binary', from_logits=True),
        }
    
    def preprocess_batch_imgs(self, imgs):
        # batch_imgs preprocessing for super images
        super_images, videos = imgs
        return super_images
    
    def postprocess_batch_preds_and_targets(self, preds, targets):
        out_preds  = []
        out_labels = []
        
        if len(preds) != len(targets['filename']):
            raise ValueError("The number of predictions and the number of targets are different ...")
        
        for i in range(len(preds)):
            pred = preds[i]
            
            trace_mask = targets['trace_mask'][i][None, :]
            pos_trace_frame = self.__get_pos_frame(targets['pos_trace_frame'], i)
            
            # Change from the channel-first into the channel-last format
            pred = pred.permute((1, 2, 0))
            
            pred_trace = get_crop_from_coors(pred, pos_trace_frame)
            
            # Change from the channel-last into the channel-first format
            pred_trace = pred_trace.permute((2,0,1))
            
            out_preds.extend([pred_trace[None, :]])
            out_labels.extend([trace_mask])
        
        out_preds  = torch.cat(out_preds)[:, :, :112, :112].contiguous()
        out_labels = torch.cat(out_labels)[:, :, :112, :112].contiguous()
        
        return out_preds, out_labels
    
    @staticmethod
    def __get_pos_frame(tensor_pos_frame, index):
        """
        Convert from
        [
            [
                tensor([224,   0, 112, 224, 112, 112, 224, 336,   0, 112,   0, 224, 336,   0,112, 112], device='cuda:0'),
                tensor([560,   0,   0, 336, 224,   0, 336,   0, 224, 336, 336, 560,   0, 560, 336, 224], device='cuda:0')
            ],
            [
                tensor([336, 112, 224, 336, 224, 224, 336, 448, 112, 224, 112, 336, 448, 112, 224, 224], device='cuda:0'),
                tensor([672, 112, 112, 448, 336, 112, 448, 112, 336, 448, 448, 672, 112, 672, 448, 336], device='cuda:0')
            ]
        ]
        
        To (for index=0) --> [[224, 560], [336, 672]]
        """
        
        tl = tensor_pos_frame[0]
        br = tensor_pos_frame[1]
        
        return [[tl[0][index], tl[1][index]],
                [br[0][index], br[1][index]]]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=3e-4,
            weight_decay=1e-5, amsgrad=True,
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[45, 60], gamma=0.1,
        )
        
        return [optimizer], [scheduler]