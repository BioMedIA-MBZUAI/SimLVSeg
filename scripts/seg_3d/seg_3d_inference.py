import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import argparse
import cv2
import numpy as np
import random
import torch
import torchvision

from tqdm import tqdm

from torch.utils.data import DataLoader

from simlvseg.utils import set_seed, load_video, save_video
from simlvseg.seg_3d.pl_module import Seg3DModule
from simlvseg.seg_3d.preprocessing import get_preprocessing_for_training

def parse_args():
    parser = argparse.ArgumentParser(description="Weakly Supervised Video Segmentation Training with 3D Models")

    parser.add_argument('--seed', type=int, default=42)

    # Paths and dataset related arguments
    parser.add_argument('--video_path', type=str, help="Path to the video", required=True)
    parser.add_argument('--mean', type=float, nargs=3, default=(0.12741163, 0.1279413, 0.12912785),
                        help="Mean normalization value (can be a list or tuple)")
    parser.add_argument('--std', type=float, nargs=3, default=(0.19557191, 0.19562256, 0.1965878),
                        help="Standard deviation normalization value (can be a list or tuple)")

    # Checkpointing arguments
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint file (can be None or a string)")
    
    # Model and training related arguments
    parser.add_argument('--encoder', type=str, default='3d_unet', help="Encoder type")
    parser.add_argument('--frames', type=int, default=32, help="Number of frames")
    parser.add_argument('--period', type=int, default=1, help="Period")

    # DataLoader arguments
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation")

    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    return args


class InferenceDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        vid_path,
        length,
        period,
        preprocessing,
    ):
        self.vid_path = vid_path
        self.video = load_video(vid_path).transpose((3, 0, 1, 2))
        self.fps   = cv2.VideoCapture(vid_path).get(cv2.CAP_PROP_FPS)
        
        self.length = length
        self.period = period
        self.preprocessing = preprocessing
        
        c, f, h, w = self.video.shape
        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            self.video = np.concatenate((self.video, np.zeros((c, length * self.period - f, h, w), self.video.dtype)), axis=1)
            c, f, h, w = self.video.shape  # pylint: disable=E0633
        
        # self.list_selected_frame_indexes = []
        # start = 0
        # while True:
        #     selected_frame_indexes = start + self.period * np.arange(length)
        #     self.list_selected_frame_indexes.append(selected_frame_indexes)
            
        #     if start == f - (length - 1) - 1:
        #         break
            
        #     start = min(start + length, f - (length - 1) - 1)
        
        self.list_selected_frame_indexes = []
        pointer = 0
        inner_loop = True
        while True:
            for i in range(self.period):
                if not inner_loop:
                    break
                
                start = pointer + i
                
                selected_frame_indexes = start + self.period * np.arange(length)
                
                if selected_frame_indexes[-1] >= f:
                    inner_loop = False
                    break
                
                self.list_selected_frame_indexes.append(selected_frame_indexes)
            
            if self.list_selected_frame_indexes[-1][-1] == f - 1:
                # print(self.list_selected_frame_indexes)
                break
            
            pointer = min(pointer + length * self.period, f - (length - 1) * self.period - self.period)
        
        print(self.list_selected_frame_indexes)
    
    def __len__(self):
        return len(self.list_selected_frame_indexes)
    
    def __getitem__(self, idx):
        selected_frame_indexes = self.list_selected_frame_indexes[idx]
        
        video = self.video[:, selected_frame_indexes, :, :].copy()
        video = video.transpose((1, 2, 3, 0))
        
        video, _ = self.preprocessing(video)
        
        # (N, C, H, W) --> (C, H, W, N)
        video = video.transpose((1, 2, 3, 0))
        
        return video, selected_frame_indexes

if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    preprocessing = get_preprocessing_for_training(
        args.frames,
        args.mean,
        args.std,
    )

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    model = Seg3DModule(args.encoder, None, 'encoder')

    device = torch.device("cuda")

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    dataset = InferenceDataset(args.video_path, args.frames, args.period, preprocessing)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    video = np.zeros_like(dataset.video[0], dtype=np.uint8)

    with torch.no_grad():
        for data in tqdm(dataloader):
            outs = model(data[0].to(device))
            
            batch_preds = outs.detach().cpu().numpy()
            batch_preds = batch_preds[:,0]
            batch_preds[batch_preds >= 0] = 255
            batch_preds[batch_preds <  0] = 0
            batch_preds = batch_preds.astype(np.uint8)
            
            for preds, frame_indexes in zip(batch_preds, data[1]):
                preds = preds.transpose(2,0,1)
                video[frame_indexes] = preds.copy()

    save_video(video, args.save_path, dataset.fps)