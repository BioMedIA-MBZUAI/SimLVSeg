import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import argparse
import os
import cv2
import numpy as np
import pytorch_lightning as pl
import random
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from simlvseg.utils import set_seed
from simlvseg.seg_3d.camus import CAMUSDatasetTest
from simlvseg.seg_3d.pl_module import Seg3DModule
from simlvseg.seg_3d.preprocessing import get_preprocessing_for_training

def parse_args():
    parser = argparse.ArgumentParser(description="Weakly Supervised Video Segmentation Training with 3D Models")

    parser.add_argument('--seed', type=int, default=42)

    # Paths and dataset related arguments
    parser.add_argument('--data_path', type=str, help="Path to the dataset", required=True)
    parser.add_argument('--mean', type=float, nargs=3, default=(0.12741163, 0.1279413, 0.12912785),
                        help="Mean normalization value (can be a list or tuple)")
    parser.add_argument('--std', type=float, nargs=3, default=(0.19557191, 0.19562256, 0.1965878),
                        help="Standard deviation normalization value (can be a list or tuple)")

    # Model and training related arguments
    parser.add_argument('--encoder', type=str, default='3d_unet', help="Encoder type")
    parser.add_argument('--frames', type=int, default=32, help="Number of frames")
    parser.add_argument('--period', type=int, default=1, help="Period")
    
    # DataLoader arguments
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation")

    # Checkpointing arguments
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint file (can be None or a string)")
    
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    preprocessing = get_preprocessing_for_training(
        args.frames,
        args.mean,
        args.std,
    )

    test_dataset = CAMUSDatasetTest(
        args.data_path,
        args.frames,
        args.mean,
        args.std,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = Seg3DModule(args.encoder, None, None)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    model.cuda()

    with torch.no_grad():
        list_preds    = []
        list_patient = []
        list_gt = []
        
        for data in tqdm(test_dataloader):
            preds = model(data[0].cuda())
            
            preds = torch.nn.Sigmoid()(preds).detach().cpu().numpy()
            
            preds = np.where(preds >= 0.5, 255, 0).astype(np.uint8)
            
            list_patient.extend(data[2])
            list_preds.extend([pred for pred in preds])
            list_gt.extend(data[1].numpy())

    for i in tqdm(range(len(list_preds))):
        patient = list_patient[i]
        
        
        if len(list_preds[i].shape) == 4:
            list_preds[i] = list_preds[i][0]
        
        list_preds[i] = list_preds[i][..., list_gt[i].sum(axis=(0,1))>0]

        path = os.path.join(args.save_dir, f'{patient}_pred.npy')
        np.save(path, list_preds[i])
