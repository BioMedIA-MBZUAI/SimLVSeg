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
        patients = [filename.split('_')[0] for filename in os.listdir(self.gt_dir) if '_gt.npy' not in filename]

        list_quality = [q.lower() for q in list_quality]
        print(list_quality)

        self.patients = []
        for patient in patients:
            qual = np.load(os.path.join(self.gt_dir, f'{patient}_quality.npy'))
            if np.char.lower(qual) not in list_quality:
                continue
            self.patients.append(patient)

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        gt  = np.load(os.path.join(self.gt_dir, f'{patient}_a4c_gt.npy'))
        pred = np.load(os.path.join(self.pred_dir, f'{patient}_pred.npy'))

        # Figure out which one is larger
        large_idx = 0 if (np.sum(gt[0] > 0) > np.sum(gt[-1] > 0)) else -1
        small_idx = 0 if (large_idx == -1) else -1
        mid_idx   = gt.shape[0] // 2

        if large_idx == small_idx:
            raise ValueError
        
        large_trace = gt[large_idx]
        small_trace = gt[small_idx]
        mid_trace   = gt[mid_idx]

        large_pred = pred[...,large_idx]
        small_pred = pred[...,small_idx]
        mid_pred   = pred[...,mid_idx]

        # print(large_trace.shape, small_trace.shape, large_pred.shape, small_pred.shape)

        large_trace = np.float32(large_trace > 0)
        small_trace = np.float32(small_trace > 0)
        mid_trace   = np.float32(mid_trace > 0)
        large_pred  = np.float32(large_pred  > 0)
        small_pred  = np.float32(small_pred  > 0)
        mid_pred    = np.float32(mid_pred  > 0)

        return large_trace, small_trace, mid_trace, large_pred, small_pred, mid_pred

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
            
            large_inter, large_union, small_inter, small_union, mid_inter, mid_union = self.run_epoch(dataloader)

            overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
            large_dice = 2 * large_inter / (large_union + large_inter)
            small_dice = 2 * small_inter / (small_union + small_inter)
            mid_dice   = 2 * mid_inter / (mid_union + mid_inter)
            
            with open(os.path.join(output_dir, "{}_dice.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, overall, large, small, mid) in zip(dataset.patients, overall_dice, large_dice, small_dice, mid_dice):
                    g.write("{},{},{},{},{}\n".format(filename, overall, large, small, mid))
            
            with open(os.path.join(output_dir, "log.csv"), "w") as f:
                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), self.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(large_inter, large_union, self.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(small_inter, small_union, self.dice_similarity_coefficient)))
                f.write("{} dice (mid)  :   {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(mid_inter, mid_union, self.dice_similarity_coefficient)))
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
        small_inter = 0
        small_union = 0
        mid_inter   = 0
        mid_union   = 0
        large_inter_list = []
        large_union_list = []
        small_inter_list = []
        small_union_list = []
        mid_inter_list   = []
        mid_union_list   = []

        for large_trace, small_trace, mid_trace, large_pred, small_pred, mid_pred in tqdm.tqdm(dataloader):
            # Count number of pixels in/out of human segmentation
            pos += (large_trace == 1).sum().item()
            pos += (small_trace == 1).sum().item()
            neg += (large_trace == 0).sum().item()
            neg += (small_trace == 0).sum().item()

            # Count number of pixels in/out of computer segmentation
            pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
            pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
            neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
            neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()
            
            y_large = large_pred

            # Compute pixel intersection and union between human and computer segmentations
            large_inter += np.logical_and(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            large_union += np.logical_or(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            large_inter_list.extend(np.logical_and(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
            large_union_list.extend(np.logical_or(y_large.detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

            y_small = small_pred

            # Compute pixel intersection and union between human and computer segmentations
            small_inter += np.logical_and(y_small.detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            small_union += np.logical_or(y_small.detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            small_inter_list.extend(np.logical_and(y_small.detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
            small_union_list.extend(np.logical_or(y_small.detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

            y_mid = mid_pred

            # Compute pixel intersection and union between human and computer segmentations
            mid_inter += np.logical_and(y_mid.detach().cpu().numpy() > 0., mid_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            mid_union += np.logical_or(y_mid.detach().cpu().numpy() > 0., mid_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
            mid_inter_list.extend(np.logical_and(y_mid.detach().cpu().numpy() > 0., mid_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
            mid_union_list.extend(np.logical_or(y_mid.detach().cpu().numpy() > 0., mid_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

        large_inter_list = np.array(large_inter_list)
        large_union_list = np.array(large_union_list)
        small_inter_list = np.array(small_inter_list)
        small_union_list = np.array(small_union_list)
        mid_inter_list   = np.array(mid_inter_list)
        mid_union_list   = np.array(mid_union_list)

        return (
            large_inter_list, large_union_list,
            small_inter_list, small_union_list,
            mid_inter_list, mid_union_list,
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