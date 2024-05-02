import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import argparse
import pytorch_lightning as pl
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import math
import yaml

import collections

import cv2
import pandas

from simlvseg.utils import defaultdict_of_lists


class CAMUSDatasetEval(torch.utils.data.Dataset):
    def __init__(self, gt_dir, pred_dir, list_quality=['good', 'medium', 'poor']):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.patients = [filename.split('_')[0] for filename in os.listdir(self.gt_dir) if '_gt.npy' not in filename]
        
        list_quality = [q.lower() for q in list_quality]

        self.data = []
        for patient in tqdm.tqdm(self.patients):
            qual = np.load(os.path.join(self.gt_dir, f'{patient}_quality.npy'))
            if np.char.lower(qual) not in list_quality:
                continue
            gt  = np.load(os.path.join(self.gt_dir, f'{patient}_a4c_gt.npy'))
            pred = np.load(os.path.join(self.pred_dir, f'{patient}_pred.npy'))

            gt = np.float32(gt > 0)
            pred  = np.float32(pred  > 0)
            
            for i in range(gt.shape[0]):
                self.data.append((gt[i], pred[...,i]))
        
        # print(len(self.data))
        # print(self.data[0][0].shape, self.data[0][1].shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TestData():
    def run_test(
            self,
            data_dir, output_dir,  pred_dir,
            list_quality, num_workers=4,
        ):
        os.makedirs(output_dir, exist_ok=True)

        # Run on test
        for split in ["test"]:
            dataset = CAMUSDatasetEval(gt_dir=data_dir, pred_dir=pred_dir, list_quality=list_quality)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=num_workers,
                shuffle=False, drop_last=False,
            )
            
            large_inter, large_union = self.run_epoch(dataloader)

            large_dice = 2 * large_inter / (large_union + large_inter)
            
            with open(os.path.join(output_dir, "{}_dice.csv".format(split)), "w") as g:
                g.write("Overall\n")
                for (overall) in zip(large_dice):
                    g.write("{}\n".format(overall))
            
            with open(os.path.join(output_dir, "log.csv"), "w") as f:
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(large_inter, large_union, self.dice_similarity_coefficient)))
                f.flush()
    
    def run_epoch(self, dataloader):
        
        total = 0.
        n = 0

        pos = 0
        neg = 0
        pos_pix = 0
        neg_pix = 0

        large_inter = 0
        large_union = 0
        large_inter_list = []
        large_union_list = []

        for large_trace, large_pred in tqdm.tqdm(dataloader):
            # Count number of pixels in/out of human segmentation
            pos += (large_trace == 1).sum().item()
            neg += (large_trace == 0).sum().item()

            # Count number of pixels in/out of computer segmentation
            pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
            neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
            
            y_large = large_pred

            # Compute pixel intersection and union between human and computer segmentations
            large_inter += np.logical_and(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            large_union += np.logical_or(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            large_inter_list.extend(np.logical_and(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
            large_union_list.extend(np.logical_or(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

        large_inter_list = np.array(large_inter_list)
        large_union_list = np.array(large_union_list)

        return (
            large_inter_list, large_union_list,
        )


    def bootstrap(self, a, b, func, samples=10000):
        """Computes a bootstrapped confidence intervals for ``func(a, b)''.

        Args:
            a (array_like): first argument to `func`.
            b (array_like): second argument to `func`.
            func (callable): Function to compute confidence intervals for.
                ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
                should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
            samples (int, optional): Number of samples to compute.
                Defaults to 10000.

        Returns:
        A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile).
        """
        a = np.array(a)
        b = np.array(b)

        bootstraps = []
        for _ in range(samples):
            ind = np.random.choice(len(a), len(a))
            bootstraps.append(func(a[ind], b[ind]))
        bootstraps = sorted(bootstraps)

        return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]
    
    def dice_similarity_coefficient(self, inter, union):
        """Computes the dice similarity coefficient.

        Args:
            inter (iterable): iterable of the intersections
            union (iterable): iterable of the unions
        """
        return 2 * sum(inter) / (sum(union) + sum(inter))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process the paths for data, prediction, and output directories.")
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the EchoNet Dynamic Dataset directory.')
    parser.add_argument('--prediction_dir', type=str, required=True, 
                        help='Path to the directory where predictions will be stored.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to the output directory where results will be saved.')
    parser.add_argument('--quality', type=str, nargs=3, default=['good', 'medium', 'poor'])

    args = parser.parse_args()

    test_model = TestData()

    test_model.run_test(data_dir=args.data_dir, pred_dir=args.prediction_dir, output_dir=args.output_dir, list_quality=args.quality)