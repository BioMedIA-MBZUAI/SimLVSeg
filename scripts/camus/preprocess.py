import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import numpy as np
import os
import SimpleITK as sitk
import cv2

from tqdm import tqdm

DIR_DATABASE_NIFTI = '/home/fadillah.maani/UniLVSeg/CAMUS_public/database_nifti'
MAX_N_FRAMES = 32
TARGET_IMG_SIZE = (112,112)
SAVE_DIR = '/home/fadillah.maani/UniLVSeg/CAMUS_public/a4c_112'

os.makedirs(SAVE_DIR, exist_ok=True)

def read_cfg_to_dict(filename):
    cfg_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            # Attempt to convert numerical values
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass
            cfg_dict[key] = value
    return cfg_dict

for patient in tqdm(os.listdir(DIR_DATABASE_NIFTI)):
    dir_patient = os.path.join(DIR_DATABASE_NIFTI, patient)
    cfg_dict = read_cfg_to_dict(os.path.join(dir_patient, 'Info_4CH.cfg'))
    try:
        assert cfg_dict['ES'] == cfg_dict['NbFrame']
        assert cfg_dict['ED'] == 1
    except:
        assert cfg_dict['ED'] == cfg_dict['NbFrame']
        assert cfg_dict['ES'] == 1
    a4c_seq = os.path.join(dir_patient, f'{patient}_4CH_half_sequence.nii.gz')
    a4c_seq = sitk.GetArrayFromImage(sitk.ReadImage(a4c_seq, sitk.sitkFloat32))

    a4c_gt = os.path.join(dir_patient, f'{patient}_4CH_half_sequence_gt.nii.gz')
    a4c_gt = sitk.GetArrayFromImage(sitk.ReadImage(a4c_gt, sitk.sitkFloat32))
    a4c_gt = np.float32(a4c_gt == 1)

    assert a4c_seq.shape[0] == a4c_gt.shape[0]

    if a4c_seq.shape[0] > MAX_N_FRAMES:
        
        seq_indices = [round(i) for i in np.linspace(0, a4c_seq.shape[0] - 1, num=MAX_N_FRAMES, dtype=float)]
        # print(a4c_seq.shape[0], seq_indices)

        a4c_seq = np.array([a4c_seq[i] for i in range(a4c_seq.shape[0]) if i in seq_indices])
        a4c_gt = np.array([a4c_gt[i] for i in range(a4c_gt.shape[0]) if i in seq_indices])

        assert a4c_seq.shape[0] == a4c_gt.shape[0] == MAX_N_FRAMES

        # print(a4c_seq.shape[0], a4c_gt.shape[0], MAX_N_FRAMES)
    
    assert a4c_seq.min() == 0
    assert a4c_seq.max() <= 255
    assert a4c_gt.min() == 0
    assert a4c_gt.max() == 1

    a4c_seq = a4c_seq.astype(np.uint8)
    a4c_gt = a4c_gt.astype(np.uint8)
    a4c_seq = np.array([cv2.resize(a4c_seq[i], TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA) for i in range(a4c_seq.shape[0])])
    a4c_gt = np.array([cv2.resize(a4c_gt[i], TARGET_IMG_SIZE, interpolation=cv2.INTER_NEAREST) for i in range(a4c_gt.shape[0])])

    # np.save('x.npy', a4c_seq)
    # np.save('y.npy', a4c_gt)
    
    np.save(os.path.join(SAVE_DIR, f'{patient}_a4c_seq.npy'), a4c_seq)
    np.save(os.path.join(SAVE_DIR, f'{patient}_a4c_gt.npy'), a4c_gt)
    np.save(os.path.join(SAVE_DIR, f'{patient}_quality.npy'), np.array(cfg_dict['ImageQuality']))