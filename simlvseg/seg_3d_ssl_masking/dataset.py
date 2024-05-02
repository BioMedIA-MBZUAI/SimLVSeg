import numpy as np
import os

from ..dataset import EchoDataset
from ..utils import load_video

class Seg3DSSLMaskingDataset(EchoDataset):
    approach='2d+time'
    def __init__(
        self,
        root=None,
        split="train",
        length=16,
        period=1,
        mask_ratio=0.5,
        random_reverse=False,
        preprocessing=None,
        augmentation=None,
    ):
        super().__init__(
            root,
            split,
            length,
            period,
            random_reverse,
            preprocessing,
            augmentation
        )
        
        if mask_ratio >= 1. or mask_ratio <= 0.:
            raise ValueError("mask_ratio must be in between 0 and 1 ...")
        
        self.mask_ratio = mask_ratio
    
    def __getitem__(self, index):
        # Find filename of video
        video = os.path.join(self.root, "Videos", self.fnames[index])
        
        # Load video into np.array
        video = load_video(video).transpose((3, 0, 1, 2))
        
        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length
        
        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633
        
        # Take a random clip from the video
        start = np.random.choice(f - (length - 1) * self.period)
        selected_frame_indexes = start + self.period * np.arange(length)
        
        # Select clips from video
        video = video[:, selected_frame_indexes, :, :]
        video = video.transpose((1, 2, 3, 0))
        
        if self.random_reverse == True:
            p = np.random.uniform(0, 1)
            if p < 0.5:
                video = video[::-1]
        
        # Augmentations
        if self.augmentation is not None:
            video, _ = self.augmentation(video)
        
        # Create a masked video
        mvideo    = video.copy()
        n_masks   = int(video.shape[0] * self.mask_ratio + 0.5)
        masks_ids = np.sort(np.random.choice(mvideo.shape[0], n_masks, replace=False))
        mvideo[masks_ids] = 0
        
        # Preprocessing
        video, _  = self.preprocessing(video)
        mvideo, _ = self.preprocessing(mvideo)
        
        # (N, C, H, W) --> (C, H, W, N)
        video  = video.transpose((1, 2, 3, 0))
        mvideo = mvideo.transpose((1, 2, 3, 0))
        
        # Gather targets
        target = {
            'filename': self.fnames[index],
            'video': video,
        }
        
        return mvideo, target
    
    def __len__(self):
        return len(self.fnames)