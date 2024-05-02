import albumentations as A
import torch

from ..albumentations import VideoAlbumentations

def get_preprocessing_for_training(
    n_frames=36,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    preprocessing = A.Compose([
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