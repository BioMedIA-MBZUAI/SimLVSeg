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

import skimage.draw
import torchvision
import cv2
import pandas

from simlvseg.utils import defaultdict_of_lists


class Echo(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root,
        pred_dir=None,
        split="test",
        target_type="EF",
        external_test_location=None,
        frame_shape = (112,112)
        ):

        super().__init__(root)

        self.frame_shape = frame_shape

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.pred_dir = pred_dir
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.preds = collections.defaultdict(list)

            self.trace = collections.defaultdict(defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])
        
        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), self.frame_shape)
                mask = np.zeros(self.frame_shape, np.float32)
                mask[r, c] = 1
                target.append(mask)
            elif t in ["LargePred", "SmallPred"]:
                #FIXME: Something off with naming convention
                if t == "LargePred":
                    mask = self.get_pred(key, self.frames[key][-1], phase='esv')
                else:
                    mask = self.get_pred(key, self.frames[key][0], phase='edv')   
            
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
        
        return target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
    
    def get_pred(self, video_name, frame, phase='edv'):
        if self.pred_dir is None:
            return None
        else:
            filename = os.path.join(self.root, self.pred_dir, "{}_{}_{}.png".format(video_name.strip('.avi'), frame, phase))
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img.astype(np.float32)/255.0


class TestData():
    def run_test(
            self,
            data_dir, output_dir,  pred_dir,
            num_workers=4,
        ):
        os.makedirs(output_dir, exist_ok=True)

        # Run on test
        for split in ["test"]:
            dataset = Echo(root=data_dir, pred_dir=pred_dir, split=split,
            target_type=["LargeTrace", "SmallTrace", "LargePred", "SmallPred"])
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=num_workers,
                shuffle=False, drop_last=False,
            )
            
            large_inter, large_union, small_inter, small_union = self.run_epoch(dataloader)

            overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
            large_dice = 2 * large_inter / (large_union + large_inter)
            small_dice = 2 * small_inter / (small_union + small_inter)
            
            with open(os.path.join(output_dir, "{}_dice.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                    g.write("{},{},{},{}\n".format(filename, overall, large, small))
            
            with open(os.path.join(output_dir, "log.csv"), "w") as f:
                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), self.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(large_inter, large_union, self.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *self.bootstrap(small_inter, small_union, self.dice_similarity_coefficient)))
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
        large_inter_list = []
        large_union_list = []
        small_inter_list = []
        small_union_list = []

        for large_trace, small_trace, large_pred, small_pred in tqdm.tqdm(dataloader):
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

        large_inter_list = np.array(large_inter_list)
        large_union_list = np.array(large_union_list)
        small_inter_list = np.array(small_inter_list)
        small_union_list = np.array(small_union_list)

        return (
            large_inter_list, large_union_list,
            small_inter_list, small_union_list,
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

    args = parser.parse_args()

    test_model = TestData()

    test_model.run_test(data_dir=args.data_dir, pred_dir=args.prediction_dir, output_dir=args.output_dir)