import copy
import numpy as np
import os
import random
import skimage.draw

from ..dataset import EchoDataset
from ..utils import (
    get_optimum_set_of_frame_indexes,
    load_video,
)

class Seg3DDataset(EchoDataset):
    approach = '2d+time'
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
        if self.preprocessing is not None:
            video, aux_outs      = self.preprocessing(video, {'trace_mask': target["trace_mask"]})
            target["trace_mask"] = aux_outs['trace_mask']
        
        # (N, C, H, W) --> (C, H, W, N)
        video = video.transpose((1, 2, 3, 0))
        
        return video, target

class Seg3DDatasetTest(Seg3DDataset):
    def __init__(self, *args, **kwargs):
        self.shuffle_temporal_order = kwargs['shuffle_temporal_order']
        kwargs.pop('shuffle_temporal_order', None)
        super().__init__(
            *args,
            **kwargs,
        )
    
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
        if self.shuffle_temporal_order == True:
            seq_index = list(range(video.shape[1]))
            random.shuffle(seq_index)

            # TYPE 1
            # curr_rel_trace_index = seq_index.index(target['rel_trace_index'])
            # seq_index[target['rel_trace_index']], seq_index[curr_rel_trace_index] = seq_index[curr_rel_trace_index], seq_index[target['rel_trace_index']]

            # assert len(seq_index) == len(set(seq_index)) # to make sure we don't have duplicates
            # assert len(selected_frame_indexes) == len(seq_index)

            # TYPE 2
            target['rel_trace_index'] = seq_index.index(target['rel_trace_index'])

            tmp_vid = copy.deepcopy(video)
            for i in range(len(seq_index)):
                tmp_vid[:,i] = video[:,seq_index[i]]

            video = tmp_vid

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
        if self.preprocessing is not None:
            video, aux_outs      = self.preprocessing(video, {'trace_mask': target["trace_mask"]})
            target["trace_mask"] = aux_outs['trace_mask']
        
        # (N, C, H, W) --> (C, H, W, N)
        video = video.transpose((1, 2, 3, 0))
        
        return video, target