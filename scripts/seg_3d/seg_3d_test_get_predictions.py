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
from simlvseg.seg_3d.dataset import Seg3DDatasetTest
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
    parser.add_argument('--shuffle_temporal_order', action='store_true')
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

    test_dataset = Seg3DDatasetTest(
        args.data_path,
        "test",
        args.frames,
        args.period,
        False,
        preprocessing,
        None,
        test=True,
        shuffle_temporal_order=args.shuffle_temporal_order,
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
        list_filename = []
        list_index    = []
        list_trace_name = []
        
        for data in tqdm(test_dataloader):
            super_images = model.preprocess_batch_imgs(data[0]).cuda()
            
            preds = model(super_images)
            
            preds, _ = model.postprocess_batch_preds_and_targets(preds, data[1])
            
            preds = torch.nn.Sigmoid()(preds).detach().cpu().numpy()
            
            preds = np.where(preds >= 0.5, 255, 0).astype(np.uint8)
            
            list_index.extend(data[1]['trace_index'].tolist())
            list_filename.extend(data[1]['filename'])
            list_preds.extend([pred for pred in preds])
            list_trace_name.extend(data[1]['trace_name'])

    for i in range(len(list_preds)):
        filename = list_filename[i]
        index = list_index[i]
        trace_name = list_trace_name[i]
        
        filename = filename.split('.')[0]
        trace_name = 'edv' if trace_name == 'small_trace' else 'esv'
        filename = f'{filename}_{index}_{trace_name}.png'
        filename = os.path.join(args.save_dir, filename)
        
        pred = list_preds[i][0]
        
        cv2.imwrite(filename, pred)
