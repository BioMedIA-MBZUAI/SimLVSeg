import numpy as np
import os
import copy
import torch

class CAMUSDatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_frames, mean, std):
        self.data_dir = data_dir
        self.n_frames = n_frames

        self.patients = sorted([filename.split('_')[0] for filename in os.listdir(self.data_dir) if '_gt.npy' not in filename])

        self.mean = np.array(mean)
        self.std  = np.array(std)
    
    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]

        a4c_seq = np.load(os.path.join(self.data_dir, f'{patient}_a4c_seq.npy'))
        a4c_gt  = np.load(os.path.join(self.data_dir, f'{patient}_a4c_gt.npy'))

        a4c_seq = np.float32(a4c_seq) / 255.
        a4c_gt  = np.float32(a4c_gt)

        if len(a4c_seq.shape) == 3:
            a4c_seq = a4c_seq[..., np.newaxis] * np.ones((1, 1, 1, 3))
        
        a4c_seq = (a4c_seq - self.mean) / self.std

        if self.n_frames > a4c_seq.shape[0]:
            # TYPE 1: MIRRORING
            # a4c_seq = expand_and_mirror(a4c_seq, self.n_frames)
            # tmp = np.zeros_like(a4c_seq)[...,0]
            # tmp[:a4c_gt.shape[0]] = a4c_gt
            # a4c_gt = tmp.copy()
            
            # TYPE 2: PADDING TO ZERO
            tmp = np.zeros((self.n_frames, 112, 112, 3)).astype(a4c_seq.dtype)
            tmp[:a4c_seq.shape[0]] = a4c_seq
            a4c_seq = tmp.copy()
            tmp = np.zeros_like(a4c_seq)[...,0]
            tmp[:a4c_gt.shape[0]] = a4c_gt
            a4c_gt = tmp.copy()

            # TYPE 3: 
            # a4c_seq = pad_array(a4c_seq, self.n_frames)
            # a4c_gt  = pad_array(a4c_gt,  self.n_frames)

            # TYPE 4: 
            # a4c_seq = pad_array_with_images(a4c_seq, self.n_frames)
            # a4c_gt  = pad_array(a4c_gt,  self.n_frames)
        
        assert a4c_seq.shape[0] == self.n_frames

        # (N, H, W, C) --> (C, H, W, N)
        a4c_seq = a4c_seq.transpose((3, 1, 2, 0))
        a4c_gt = a4c_gt.transpose((1,2,0))

        a4c_seq = np.float32(a4c_seq)
        a4c_gt  = np.float32(a4c_gt)

        return a4c_seq, a4c_gt, patient

def expand_and_mirror(X, M):
    N, H, W, C = X.shape
    
    assert M > N
    
    # Calculate the total number of repetitions needed to exceed or meet M
    repeat_factor = (M + N - 1) // N # Ceiling division to ensure at least M elements
    
    # Generate a mirrored sequence of indices
    mirrored_indices = np.arange(N)
    for _ in range(repeat_factor // 2):
        mirrored_indices = np.concatenate([mirrored_indices, mirrored_indices[::-1]])
    
    # If the repeat factor is odd, add one more direct copy of the original indices
    if repeat_factor % 2 != 0:
        mirrored_indices = np.concatenate([mirrored_indices, np.arange(N)])
    
    # Ensure the mirrored sequence is at least of length M and truncate if necessary
    mirrored_indices = mirrored_indices[:M]
    
    # Use advanced indexing to create the expanded and mirrored array
    expanded_X = X[mirrored_indices]
    
    return expanded_X

def pad_array(X, M):
    """
    Pad the numpy array X of shape (N, H, W, 3) to a new shape (M, H, W, 3)
    by adding padding on both edges of the first dimension.
    
    Parameters:
    - X: Input array of shape (N, H, W, 3)
    - M: The new size of the first dimension after padding
    
    Returns:
    - Padded array of shape (M, H, W, 3)
    """
    N, H, W = X.shape[:3]  # Original dimensions of X
    
    # Calculate the total padding needed
    total_pad = M - N
    # Ensure that the total padding is non-negative
    if total_pad < 0:
        raise ValueError("M must be greater than N")
    
    # Calculate padding for the beginning and end of the first dimension
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    
    # Create padding configuration
    if len(X.shape) == 4:
        pad_width = [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)]
    elif len(X.shape) == 3:
        pad_width = [(pad_before, pad_after), (0, 0), (0, 0)]
    
    # Apply padding
    padded_X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded_X

import numpy as np

def pad_array_with_images(X, M):
    """
    Pad the numpy array X of shape (N, H, W, 3) to a new shape (M, H, W, 3)
    by adding the first and last image on both edges of the first dimension.
    
    Parameters:
    - X: Input array of shape (N, H, W, 3)
    - M: The new size of the first dimension after padding
    
    Returns:
    - Padded array of shape (M, H, W, 3)
    """
    N, H, W, C = X.shape  # Original dimensions of X
    
    # Calculate the total padding needed
    total_pad = M - N
    # Ensure that the total padding is non-negative
    if total_pad < 0:
        raise ValueError("M must be greater than N")
    
    # Calculate padding for the beginning and end of the first dimension
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    
    # Replicate the first and last image for padding
    pad_before_images = np.repeat(X[:1], pad_before, axis=0)
    pad_after_images = np.repeat(X[-1:], pad_after, axis=0)
    
    # Concatenate the padding and the original array
    padded_X = np.concatenate([pad_before_images, X, pad_after_images], axis=0)
    
    return padded_X