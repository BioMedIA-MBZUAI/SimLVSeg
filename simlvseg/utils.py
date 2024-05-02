import os
import cv2
import collections
import numpy as np
import random
import torch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """
    
    return collections.defaultdict(list)

def get_crop_from_coors(img, tl_br_coor):
    tl_coor, br_coor = tl_br_coor
    return img[tl_coor[0]:br_coor[0], tl_coor[1]:br_coor[1]]

def get_tl_br_coors(size, n_grid):
    # Get top-left and bottom-right coordinates
    # of each grid
    step = size // n_grid
    
    if step != size / n_grid:
        raise ValueError('abcdefghijklmn')
    
    tl_br_coors = []
    for i in range(n_grid):
        tl_br_coors.extend([[[i*step, j*step], [(i+1)*step, (j+1)*step]] for j in range(n_grid)])
    
    return tl_br_coors

def get_optimum_set_of_frame_indexes(length, period, target_index, n_frames):
    candidates = [target_index]
    for i in range(length - 1):
        if i%2 == 0:
            candidates.insert(0, candidates[0] - period)
        else:
            candidates.append(candidates[-1] + period)
    
    candidates = [i for i in candidates if i < n_frames]
    
    while len(candidates) < length:
        candidates.insert(0, candidates[0] - period)
    
    selected_frames = []
    for candidate in reversed(candidates):
        if candidate < 0:
            selected_frames.append(selected_frames[-1] + period)
        else:
            selected_frames.insert(0, candidate)
    
    return selected_frames

def load_video(filename: str) -> np.ndarray:
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)
    
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)
    
    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame
    
    return v

def save_video(video, save_path, fps=10):
    img = video[0]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (img.shape[0], img.shape[1]))
    for img in video:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        out.write(img)
    out.release()