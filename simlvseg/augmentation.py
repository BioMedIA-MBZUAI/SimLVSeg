import albumentations as A
import cv2

from .albumentations import VideoAlbumentations

def get_augmentation(
    n_frames=36,
):
    augmentation = A.Compose([
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
        A.CLAHE(p=0.5),
        A.Rotate(20, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT, value=0, p=1.),
        A.PadIfNeeded(124, 124, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.RandomCrop(112, 112, always_apply=True),
    ])
    
    augmentation = VideoAlbumentations(n_frames, augmentation, {'trace_mask': 'mask'})
    
    return augmentation