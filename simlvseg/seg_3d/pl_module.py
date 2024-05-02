import torch

from ..pl_module import SegModule

class Seg3DModule(SegModule):
    def preprocess_batch_imgs(self, imgs):
        return imgs
    
    def postprocess_batch_preds_and_targets(self, preds, targets):
        out_preds  = []
        out_labels = []
        
        if len(preds) != len(targets['filename']):
            raise ValueError("The number of predictions and the number of targets are different ...")
        
        for i in range(len(preds)):
            pred = preds[i]
            
            trace_mask = targets['trace_mask'][i][None, :]
            pred_trace = pred[..., targets['rel_trace_index'][i]]
            
            out_preds.extend([pred_trace[None, :]])
            out_labels.extend([trace_mask])
        
        out_preds  = torch.cat(out_preds)
        out_labels = torch.cat(out_labels)
        
        return out_preds, out_labels