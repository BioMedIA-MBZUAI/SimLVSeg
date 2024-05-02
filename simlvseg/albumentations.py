import albumentations as A
import numpy as np

class VideoAlbumentations:
    def __init__(
        self,
        n_frames,
        transform,
        aux_targets=None,
    ):
        aux_targets = aux_targets if aux_targets is not None else {}
        if (aux_targets is not None) and not isinstance(aux_targets, dict):
            raise ValueError("aux_targets must be a dictionary ...")
        
        self.n_frames = n_frames
        
        self.transform = transform
        self.transform.add_targets({
            **{f'image{i}': 'image' for i in range(self.n_frames)},
            **aux_targets
        })
    
    def __call__(
        self,
        video,
        aux_inputs=None,
    ):
        if len(video) != self.n_frames:
            raise ValueError(f'Get {len(video)} frames but we expect {self.n_frames} frames ...')
        
        inputs = {f'image{i}': video[i] for i in range(self.n_frames)}
        inputs = {**inputs, **aux_inputs} if isinstance(aux_inputs, dict) else inputs
        inputs['image'] = video[0]
        
        outs = self.transform(**inputs)
        
        out_video = [outs[f'image{i}'][None, :] for i in range(self.n_frames)]
        out_video = np.concatenate(out_video, axis=0)
        
        out_auxs = {key: outs[key] for key in aux_inputs.keys()} \
            if isinstance(aux_inputs, dict) else None
        
        return out_video, out_auxs