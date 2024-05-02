# Reference
# https://github.com/echonet/dynamic/blob/master/echonet/utils/__init__.py#L125-L138

"""EchoNet-Dynamic Dataset."""

import os
import collections
import cv2
import einops
import math
import pandas
import random

import numpy as np
import skimage.draw
import torchvision

from .utils import (
    defaultdict_of_lists,
    get_tl_br_coors,
    get_optimum_set_of_frame_indexes,
    load_video,
)

class EchoDataset(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.
    
    Args:
        root (string): Root directory of dataset (defaults to None)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
    """
    approach = 'super_image'
    def __init__(
        self,
        root=None,
        split="train",
        length=16,
        period=1,
        random_reverse=False,
        preprocessing=None,
        augmentation=None,
        test=False,
        pct_train=None,
    ):
        if root is None:
            raise ValueError("root cannot be None")
        
        super().__init__(root)
        
        self.split = split.upper()
        self.length = length
        self.period = period
        self.random_reverse = random_reverse
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.test = test
        
        self.n_grid = int(math.sqrt(self.length))
        if self.n_grid**2 != self.length and self.approach=='super_image':
            raise ValueError("Hihihihihihi")
        
        self.fnames, self.outcome = [], []
        
        # Load video-level labels
        with open(os.path.join(self.root, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        data["Split"].map(lambda x: x.upper())
        
        if self.split != "ALL":
            data = data[data["Split"] == self.split]
        
        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()
        self.fnames = [fn + ".avi" for fn in self.fnames if (os.path.splitext(fn)[1] == "")]  # Assume avi if no suffix
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
        
        if pct_train is not None:
            random.seed(42)
            self.fnames = random.sample(self.fnames, int(len(self.fnames)*pct_train + 0.5))
        
        self.data_map = []
        for fname_index in range(len(self.fnames)):
            for trace_name in ['large_trace', 'small_trace']:
                self.data_map.append([fname_index, trace_name])
    
    def __getitem__(self, idx):
        # Get fname index and trace name
        index, trace_name = self.data_map[idx]
        
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
        
        # Gather targets
        target = {}
        
        key = self.fnames[index]
        _t  = -1 if trace_name == 'large_trace' else 0
        
        target['filename']    = self.fnames[index]
        target['trace_name']  = trace_name
        target['trace_index'] = np.int(self.frames[key][_t])
        target['trace_frame'] = video[:, self.frames[key][_t], :, :].transpose((1, 2, 0))
        t = self.trace[key][self.frames[key][_t]]
        
        x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))
        
        r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
        mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
        mask[r, c] = 1
        
        target['trace_mask'] = mask[:, :, None]
        
        # Take a random clip from the video
        if self.test is False:
            while True:
                start = np.random.choice(f - (length - 1) * self.period)
                selected_frame_indexes = start + self.period * np.arange(length)
                
                if (target['trace_index'] >= selected_frame_indexes[0]) and \
                    (target['trace_index'] <= selected_frame_indexes[-1]):
                    if target['trace_index'] in selected_frame_indexes:
                        break
        else:
            selected_frame_indexes = get_optimum_set_of_frame_indexes(
                length, self.period, target['trace_index'], f
            )
            start = selected_frame_indexes[0]
        
        target['selected_frame_indexes'] = selected_frame_indexes
        target['rel_trace_index']        = (target['trace_index'] - start) // self.period
        
        # Select clips from video
        video = video[:, selected_frame_indexes, :, :]
        video = video.transpose((1, 2, 3, 0))
        
        if self.random_reverse == True:
            raise NotImplementedError("We have not implemented a code to reverse the index ...")
            p = np.random.uniform(0, 1)
            if p < 0.5:
                video = video[::-1]
        
        # Augmentations
        if self.augmentation is not None:
            video, aux_outs      = self.augmentation(video, {'trace_mask': target["trace_mask"]})
            target["trace_mask"] = aux_outs['trace_mask']
        
        # Preprocessing
        video, aux_outs      = self.preprocessing(video, {'trace_mask': target["trace_mask"]})
        target["trace_mask"] = aux_outs['trace_mask']
        
        # Create a super image
        si = einops.rearrange(video, '(nh nw) c h w -> c (nh h) (nw w)',
                              nh=self.n_grid, nw=self.n_grid)
        
        # Get the coordinates of large and small frames
        # Only works for square images or square grids
        if si.shape[1] != si.shape[2]:
            raise NotImplementedError("Currently this only works for images having square shape ...")
        tl_br_coors = get_tl_br_coors(si.shape[1], self.n_grid)
        target['pos_trace_frame'] = tl_br_coors[target['rel_trace_index']]
        
        return (si, video), target
    
    def __len__(self):
        return len(self.data_map)
    
    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)