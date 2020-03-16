import torchvision.transforms._transforms_video as transforms
import torchvision.transforms._functional_video as F
import torch
import numpy as np
from PIL import Image


class ToTensorVideo:
    def __call__(self, clip):
        """
        :param clip: T x C x H x W list of PIL images
        :return: C x T x H x W uint8 tensor
        """
        assert isinstance(clip[0], Image.Image), 'should be a Pil image'
        clip = [torch.from_numpy(np.array(frame)) for frame in clip]  # T x C x H x W
        clip = torch.stack(clip, dim=0).float().permute(dims=(3, 0, 1, 2))  # C x T x H x W
        return clip

class ToZeroOneVideo:
    def __call__(self, clip):
        """
        :param clip: C x T x H x W tensor
        :return: C x T x H x W float32 tensor between 0 and 1
        """
        assert isinstance(clip, torch.Tensor)
        return clip.float() / 255.0


class ResizeVideo:
    def __init__(self, size, interpolation_mode='bilinear'):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        assert F._is_tensor_video_clip(clip)
        clip = F.resize(clip, self.size, interpolation_mode=self.interpolation_mode)
        return clip
