import numpy as np
import numbers
import torch
import torchvision.transforms.functional as F
import cv2
import random
from PIL import Image
import torchvision.transforms as imgtransforms


INTERPOLATION = {
    'bilinear': {
        'cv2': cv2.INTER_LINEAR,
        'pil': Image.BILINEAR
    }
}


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def get_clip_frame_size(clip):
    if isinstance(clip[0], np.ndarray):
        im_h, im_w, _ = clip[0].shape
    elif isinstance(clip[0], Image.Image):
        im_w, im_h = clip[0].size
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return im_h, im_w


def _get_resize_size(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def _resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = _get_resize_size(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = _get_resize_size(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = Image.NEAREST
        else:
            pil_inter = Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return scaled


def _crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


def _horizontal_flip(clip):
    if isinstance(clip[0], np.ndarray):
        flipped_clip = [np.fliplr(img) for img in clip]
    elif isinstance(clip[0], Image.Image):
        flipped_clip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return flipped_clip


class ToTensor(object):
    """Converts a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    or PIL Images to a torch.FloatTensor of shape (m*C x H x W)
    in the range [0, 1.0]
    """

    def __init__(self, max_val=255):
        self.max_val = float(max_val)

    def __call__(self, clip):
        """ Args: clip (list of numpy.ndarray or PIL.Image.Image): clip (list of images) to be converted to tensor. """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            pass
        elif isinstance(clip[0], Image.Image):
            clip = [np.array(frame) for frame in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))

        clip = [np.transpose(frame, axes=[2, 0, 1]) for frame in clip]
        clip = np.array(clip).astype(np.float32) / self.max_val
        clip = torch.from_numpy(clip)
        return clip


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        assert 0 <= flip_prob <= 1
        self.flip_prob = float(flip_prob)

    def __call__(self, clip):
        """
        Args:  img (PIL.Image or numpy.ndarray): List of images to be cropped in format (h, w, c) in numpy.ndarray
        Returns: PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.flip_prob:
            clip = _horizontal_flip(clip)
        return clip


class RandomResize:
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (width, height)
    """
    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])
        im_h, im_w = get_clip_frame_size(clip)

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = _resize_clip(clip, new_size, interpolation=self.interpolation)
        return resized


class Rescale:
    """ This is an aspect preserving resize """
    def __int__(self, min_resize, interpolation='nearest'):
        self.min_resize = int(min_resize)
        self.interpolation = interpolation

    def __call__(self, clip):
        im_h, im_w = get_clip_frame_size(clip)
        if im_h > im_w:
            new_w = self.min_resize
            new_h = self.min_resize / im_w * im_h
        else:
            new_h = self.min_resize
            new_w = self.min_resize / im_h * im_w
        new_size = (new_w, new_h)
        return _resize_clip(clip, new_size, interpolation=self.interpolation)


class Resize:
    """
    Resizes a list of (H x W x C) numpy.ndarray to the final size
    size (tuple): (widht, height)
    """
    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        return _resize_clip(clip, self.size, interpolation=self.interpolation)


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = ('Initial image size should be larger then cropped size but got cropped sizes : ({w}, {h}) '
                         'while initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = _crop_clip(clip, y1, x1, h, w)
        return cropped


class ToPil:
    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            out = [Image.fromarray(frame) for frame in clip]
        elif isinstance(clip[0], Image.Image):
            out = clip
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
        return out


class ColorJitter:
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        to_cv2_array = False
        if isinstance(clip[0], np.ndarray):
            to_cv2_array = True
            clip = [Image.fromarray(frame) for frame in clip]
        elif isinstance(clip[0], Image.Image):
            pass
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))

        brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation,
                                                                self.hue)

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda image: F.adjust_brightness(image, brightness))
        if saturation is not None:
            img_transforms.append(lambda image: F.adjust_saturation(image, saturation))
        if hue is not None:
            img_transforms.append(lambda image: F.adjust_hue(image, hue))
        if contrast is not None:
            img_transforms.append(lambda image: F.adjust_contrast(image, contrast))
        random.shuffle(img_transforms)

        # Apply to all images
        jittered_clip = []
        for img in clip:
            for func in img_transforms:
                img = func(img)
            jittered_clip.append(img)

        if to_cv2_array:
            jittered_clip = [np.array(frame) for frame in jittered_clip]
        return jittered_clip


class RandomRescaledCrop:
    def __init__(self, size, scale=(256, 320)):
        self.size = size
        self.scale = scale

    def __call__(self, img_group):
        shortedge = float(random.randint(*self.scale))

        w, h, _ = img_group[0].shape
        scale = max(shortedge / w, shortedge / h)
        img_group = [mmcv.imrescale(img, scale) for img in img_group]
        w, h, _ = img_group[0].shape
        w_offset = random.randint(0, w - self.size[0])
        h_offset = random.randint(0, h - self.size[1])

        box = np.array([w_offset, h_offset,
                        w_offset + self.size[0] - 1, h_offset + self.size[1] - 1],
                        dtype=np.float32)

        return ([img[w_offset: w_offset + self.size[0], h_offset: h_offset + self.size[1]] for img in img_group], box)


class RandomCrop(object):
    """ Extract random crop at the same location for a list of images """
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            crop_size = (crop_size, crop_size)
        self.size = crop_size

    def __call__(self, clip):
        """
        Args: img (PIL.Image or numpy.ndarray): List of images to be cropped in format (h, w, c) in numpy.ndarray Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        im_h, im_w = get_clip_frame_size(clip)
        if w > im_w or h > im_h:
            error_msg = ('Initial image size should be larger then cropped size but got cropped sizes : '
                         '({w}, {h}) while initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = _crop_clip(clip, min_h=y1, min_w=x1, h=h, w=w)
        return cropped


class Crop:
    """ crop at the specified dimensions """
    def __init__(self, start_w, start_h, crop_w, crop_h):
        self.start_w = int(start_w)
        self.start_h = int(start_h)
        self.crop_w = int(crop_w)
        self.crop_h = int(crop_h)

    def __call__(self, clip):
        return _crop_clip(clip, min_h=self.start_h, min_w=self.start_w, w=self.crop_w, h=self.crop_h)


class TenCrop:
    """ crops center, top-left, top-right, btm-right, btm-left and their flips"""
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, img_group, is_flow=False):

        image_h = img_group[0].shape[0]
        image_w = img_group[0].shape[1]
        crop_w, crop_h = self.crop_size

        offsets = MultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = _crop_clip(img_group, o_h, o_w, crop_h, crop_w)
            flip_group = _horizontal_flip(normal_group)
            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class ThreeCrop:
    """ crops the left, center and right or top, bottom or down """
    def __init__(self, crop_size):
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    def __call__(self, clip, is_flow=False):
        image_h, image_w = get_clip_frame_size(clip)
        crop_w, crop_h = self.crop_size
        assert crop_h == image_h or crop_w == image_w

        offsets = list()
        if crop_h == image_h:
            w_step = (image_w - crop_w) // 2
            offsets.append((0, 0))  # left
            offsets.append((2 * w_step, 0))  # right
            offsets.append((w_step, 0))  # middle
        elif crop_w == image_w:
            h_step = (image_h - crop_h) // 2
            offsets.append((0, 0))  # top
            offsets.append((0, 2 * h_step))  # down
            offsets.append((0, h_step))  # middle

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = _crop_clip(clip, o_h, o_w, crop_h, crop_w)
            oversample_group.extend(normal_group)
        return oversample_group, None


class MultiScaleCrop:
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = 'bilinear'

    def __call__(self, clip, is_flow=False):
        im_h, im_w = get_clip_frame_size(clip)

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size((im_w, im_h))
        crop_im_group = _crop_clip(clip, offset_h, offset_w, crop_h, crop_w)
        ret_img_group = _resize_clip(crop_im_group, (self.input_size[0], self.input_size[1]),
                                     interpolation=self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret
