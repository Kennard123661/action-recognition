import os
import data.breakfast as breakfast
import h5py
import torch.utils.data as tdata
from torch.utils.data._utils.collate import default_collate
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numbers
from PIL import Image
import numpy as np

from scripts import BASE_CONFIG_DIR, BASE_CHECKPOINT_DIR, BASE_LOG_DIR
from utils.video_reader import sample_video_frames
FRAME_REG_LOG_DIR = os.path.join(BASE_LOG_DIR, 'frame-recognition')
FRAME_REG_CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'frame-recognition')
FRAME_REG_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, 'frame-recognition')

RESCALE_SIZE = 256
CROP_SIZE = 224



INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]


class ComposeVideo:
    def __init__(self, video_transforms):
        self.video_transforms = video_transforms

    def __call__(self, clip):
        for video_transform in self.video_transforms:
            clip = video_transform(clip)
        return clip


class OneOfVideo:
    def __init__(self, video_transforms):
        self.video_transforms = video_transforms

    def __call__(self, clip):
        video_transform = np.random.choice(self.video_transforms)
        return video_transform(clip)


class ToPilVideo:
    def __call__(self, clip):
        assert isinstance(clip[0], np.ndarray), 'frame is type {}'.format(type(clip))
        return [Image.fromarray(frame) for frame in clip]


class ToTensorVideo:
    def __init__(self):
        self.im_transform = transforms.ToTensor()

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image), 'frame is not a PIL image'
        clip = [self.im_transform(frame) for frame in clip]
        clip = torch.stack(clip, dim=0)
        clip = clip.permute(1, 0, 2, 3)  # C x T x H x W
        return clip


class NormalizeVideo:
    def __init__(self, mean, std, inplace=False):
        assert len(mean) == len(std) == 3
        self.mean = mean
        self.std = std
        self.inplace = bool(inplace)

    def __call__(self, clip):
        assert isinstance(clip, torch.Tensor) and len(clip.shape) == 4  # should be T x C x H x W
        mean = torch.as_tensor(self.mean).view(3, 1, 1, 1)
        std = torch.as_tensor(self.std).view(3, 1, 1, 1)
        if not self.inplace:
            clip = clip.clone()
        clip.sub_(mean).div_(std)
        return clip


class RescaleVideo:
    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        """
        :param clip:
        :return: aspect-preserving resize of the frames in the clip
        """
        assert isinstance(clip[0], Image.Image)
        w, h = clip[0].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return clip
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return [img.resize((ow, oh), self.interpolation) for img in clip]
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return [img.resize((ow, oh), self.interpolation) for img in clip]


class CenterCropVideo:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple) or isinstance(size, np.ndarray) or isinstance(size, list):
            self.size = size
        else:
            raise TypeError

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return [F.crop(img, crop_top, crop_left, crop_height, crop_width) for img in clip]


class RandomHorizontalFlipVideo:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        if random.random() < self.p:
            clip = [F.hflip(frame) for frame in clip]
        return clip


class RandomVerticalFlipVideo:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        if random.random() < self.p:
            clip = [F.vflip(frame) for frame in clip]
        return clip


class RandomRotationVideo:
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        assert isinstance(degrees, float) and degrees > 0
        self.degrees = (-degrees, degrees)
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        angle = self.get_params(self.degrees)
        return [F.rotate(img, angle, self.resample, self.expand, self.center, self.fill) for img in clip]


class RandomCropVideo:
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        elif isinstance(size, tuple) or isinstance(size, np.ndarray) or isinstance(size, list):
            self.size = size
        else:
            raise TypeError('invalid type for size')
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        if self.padding is not None:
            clip = [F.pad(img, self.padding, self.fill, self.padding_mode) for img in clip]

        # pad the width if needed
        if self.pad_if_needed and clip[0].size[0] < self.size[1]:
            clip = [F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode) for img in clip]
        # pad the height if needed
        if self.pad_if_needed and clip[0].size[1] < self.size[0]:
            clip = [F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode) for img in clip]

        i, j, h, w = self.get_params(clip[0], self.size)

        return [F.crop(img, i, j, h, w) for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class ToGrayScaleVideo:
    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        return [F.to_grayscale(img, num_output_channels=3) for img in clip]


class ColorJitterVideo:
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        im_transforms = []
        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            im_transforms.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            im_transforms.append(transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            im_transforms.append(transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            im_transforms.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(im_transforms)
        im_transforms = transforms.Compose(im_transforms)
        return im_transforms

    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image)
        im_transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return [im_transform(img) for img in clip]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class FrameRecDataset(tdata.Dataset):
    def __init__(self, videos, input_size, frame_stride=1, n_classes=48, segment_length=30):
        super(FrameRecDataset, self).__init__()
        assert frame_stride >= 1, 'this should be larger than or equals to 1'
        assert (segment_length % 2) == 0
        self.frame_stride = frame_stride
        self.n_classes = n_classes
        self.segment_length = segment_length
        self.frame_size = int(input_size)
        video_files = [os.path.join(breakfast.VIDEO_DIR, video) for video in videos]
        for file in video_files:
            assert os.path.exists(file)
        video_length_files = [os.path.join(breakfast.VIDEO_LENGTHS_DIR, video + '.npy') for video in videos]
        video_lengths = [np.load(file) for file in video_length_files]
        video_label_files = [os.path.join(breakfast.FRAME_RECOGNITION_LABEL_DIR, video + '.labels.hdf5') for video in videos]
        action_to_logit_dict = breakfast.read_mapping_file()

        self.frame_idxs = []
        self.video_files = []
        self.labels = []
        self.video_lengths = video_lengths
        for i, video_file in enumerate(video_files):
            video_length = video_lengths[i]
            video_label_file = video_label_files[i]
            with h5py.File(video_label_file, 'r') as f:
                video_labels = f['labels'][...]
            video_labels = [action_to_logit_dict[label.decode('utf8')] for label in video_labels]
            for j in range(video_length):
                if j >= len(video_labels):
                    continue
                label = video_labels[j]
                if label < n_classes:
                    self.video_files.append(video_file)
                    self.frame_idxs.append(j)
                    self.labels.append(label)
                    self.video_lengths.append(video_length)

        self.transforms = ComposeVideo([
            ToPilVideo(),
            RescaleVideo(size=input_size),
            CenterCropVideo(size=input_size),
            ToTensorVideo(),
            NormalizeVideo(mean=INPUT_MEAN, std=INPUT_STD)
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        frame_idx = self.frame_idxs[idx]
        label = self.labels[idx]
        video_length = self.video_lengths[idx]

        left_idxs = np.sort(frame_idx - (np.arange(0, self.segment_length // 2) + 1) * self.frame_stride)
        right_idxs = np.sort(frame_idx + (np.arange(0, self.segment_length // 2) + 1) * self.frame_stride)
        idxs = np.concatenate([left_idxs, [frame_idx], right_idxs], axis=0)

        start_idx = np.argwhere(idxs >= 0).reshape(-1)[0]
        end_idx = np.argwhere(idxs < video_length).reshape(-1)[-1]
        sample_idxs = idxs[start_idx:end_idx+1]
        assert len(sample_idxs) > 0, 'there are zero frames to sample...'

        video_frames = sample_video_frames(video_file, sample_idxs)
        height, width, n_channels = video_frames[0].shape
        if start_idx != 0:
            left_padding_size = start_idx - 0
            left_padding = [np.zeros(shape=[height, width, n_channels]).astype(np.uint8)] * left_padding_size
            video_frames = left_padding + video_frames
        if end_idx < len(idxs) - 1:
            right_padding_size = len(idxs) - end_idx
            right_padding = [np.zeros(shape=[height, width, n_channels]).astype(np.uint8)] * right_padding_size
            video_frames += right_padding
        video_frames = self.transforms(video_frames)
        assert video_frames.shape[1] == (self.segment_length + 1)  # check length of video frames
        return video_frames, label

    @staticmethod
    def collate_fn(batch):
        video_frames, labels = zip(*batch)
        video_frames = default_collate(video_frames)
        labels = default_collate(labels)
        return video_frames, labels


def get_train_videos():
    return breakfast.get_split_videonames(split='train')


def get_test_videos():
    return breakfast.get_split_videonames(split='test')


dset = FrameRecDataset(videos=os.listdir(breakfast.VIDEO_DIR), frame_stride=2, input_size=224)
print(len(dset))
print(dset.__getitem__(0))
