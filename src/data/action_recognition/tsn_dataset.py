import torch.utils.data as tdata
import numpy as np


class TsnDataset(tdata.Dataset):
    def __init__(self, n_segments=3, new_length=1, new_step=1, random_shift=True,
                 temporal_jitter=False, modality='RGB', img_scale=256, input_size=224,
                 div_255=False, size_divsior=None, flip_ratio=0.5, resize_ratio=(1, 0.875, 0.75, 0.66),
                 test_mode=False, oversample=None, random_crop=False, more_fix_crop=False, resize_crop=False,
                 rescale_crop=False, scales=None, max_distort=1):
        super(TsnDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TrainDataset(tdata.Dataset):
    def __init__(self, video_dicts, n_segments=3, segment_len=5, n_steps=1, random_shift=True, modality='Flow', img_scale=256,
                 input_size=224, div_255=False, flip_ratio=0.5, keep_aspect_ratio=True, oversample=None,
                 random_crop=False, more_fixed_crop=False, multiscale_crop=True, scales=(1, 0.875, 0.75, 0.66),
                 max_distort=1):
        super(TrainDataset, self).__init__()

        self.n_segments = int(n_segments)
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        self.flip_ratio = float(flip_ratio)
        self.keep_aspect_ratio = keep_aspect_ratio

        assert oversample in [None, 'three_crop', 'ten_crop']




