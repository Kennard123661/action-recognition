import torch.utils.data as tdata
import numpy as np
import utils.videotransforms as videotransforms


class GroupImageTransform(object):
    """Preprocess a group of images.
    1. rescale the images to expected size
    2. (for classification networks) crop the images with a given size
    3. flip the images (if needed)
    4(a) divided by 255 (0-255 => 0-1, if needed)
    4. normalize the images
    5. pad the images (if needed)
    6. transpose to (c, h, w)
    7. stack to (N, c, h, w)
    where, N = 1 * N_oversample * N_seg * L
    """

    def __init__(self,  mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True, size_divisor=None, crop_size=None,
                 oversample=None, random_crop=False, resize_crop=False, rescale_crop=False, more_fix_crop=False,
                 multiscale_crop=False, scales=None, max_distort=1):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        self.resize_crop = resize_crop
        self.rescale_crop = rescale_crop

        self.to_tensor()

        # croping parameters
        if crop_size is not None:
            if oversample == 'three_crop':
                self.op_crop = videotransforms.ThreeCrop(crop_size)
            elif oversample == 'ten_crop':
                # oversample crop (test)
                self.op_crop = videotransforms.TenCrop(crop_size)
            elif resize_crop:
                # self.op_crop = videotransforms.Crop(crop_size)
                pass
            elif multiscale_crop:
                # multiscale crop (train)
                self.op_crop = videotransforms.MultiScaleCrop(crop_size, scales=scales, max_distort=max_distort,
                                                              fix_crop=not random_crop, more_fix_crop=more_fix_crop)
            else:
                self.op_crop = videotransforms.CenterCrop(crop_size)
        else:
            self.op_crop = None

    def __call__(self, clip, scale, crop_history=None, flip=False,
                 keep_ratio=True, div_255=False, is_flow=False):
        if self.resize_crop or self.rescale_crop:
            clip, crop_quadruple = self.op_crop(clip)
            im_h, im_w = videotransforms.get_clip_frame_size(clip)
            scale_factor = None
        else:
            # 1. rescale
            if keep_ratio:
                tuple_list = [videotransforms.imrescale(
                    img, scale, return_scale=True) for img in clip]
                clip, scale_factors = list(zip(*tuple_list))
                scale_factor = scale_factors[0]
            else:
                tuple_list = [mmcv.imresize(
                    img, scale, return_scale=True) for img in clip]
                clip, w_scales, h_scales = list(zip(*tuple_list))
                scale_factor = np.array([w_scales[0], h_scales[0],
                                         w_scales[0], h_scales[0]],
                                        dtype=np.float32)
            # 2. crop (if necessary)
            if crop_history is not None:
                self.op_crop = GroupCrop(crop_history)
            if self.op_crop is not None:
                clip, crop_quadruple = self.op_crop(
                    clip, is_flow=is_flow)
            else:
                crop_quadruple = None

            img_shape = clip[0].shape
        # 3. flip
        if flip:
            clip = [mmcv.imflip(img) for img in clip]
        if is_flow:
            for i in range(0, len(clip), 2):
                clip[i] = mmcv.iminvert(clip[i])

        # 4a. div_255
        if div_255:
            clip = videotransforms.normalize(clip, 0, 255)

        # 4. normalize
        # todo: add slef to rgb
        clip = videotransforms.normalize(clip, self.mean, self.std)

        # 5. pad
        if self.size_divisor is not None:
            clip = [mmcv.impad_to_multiple(img, self.size_divisor) for img in clip]
            pad_shape = clip[0].shape
        else:
            pad_shape = img_shape
        if is_flow:
            assert len(clip[0].shape) == 2
            clip = [np.stack((flow_x, flow_y), axis=2)
                    for flow_x, flow_y in zip(
                    clip[0::2], clip[1::2])]
        # 6. transpose
        clip = [img.transpose(2, 0, 1) for img in clip]

        # Stack into numpy.array
        clip = np.stack(clip, axis=0)
        return clip, img_shape, pad_shape, scale_factor, crop_quadruple


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




