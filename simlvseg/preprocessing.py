import albumentations as A
import cv2
import torch

from .albumentations import VideoAlbumentations

def get_preprocessing_for_training(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    n_frames=16,
):
    preprocessing = A.Compose([
        A.PadIfNeeded(128, 128, position='top_left', border_mode=cv2.BORDER_CONSTANT,
                      value=0, mask_value=0, always_apply=True),
        A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        A.Lambda(image=ToTensor(), mask=ToTensor(), always_apply=True),
    ])
    
    preprocessing = VideoAlbumentations(n_frames, preprocessing, {'trace_mask': 'mask'})
    
    return preprocessing

class ToTensor():
    def __init__(self, dtype='float32'):
        self.dtype = dtype
    
    def __call__(self, x, **kwargs):
        return torch.from_numpy(x.transpose(2, 0, 1).astype(self.dtype))