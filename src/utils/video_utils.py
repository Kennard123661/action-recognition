import torchvision.transforms._transforms_video as transforms
import torch
from PIL import Image


class ToTensor:
    def __call__(self, clip):
        """
        :param clip: T x C x H x W list of PIL images
        :return: C x T x H x W uint8 tensor
        """
        assert isinstance(clip[0], Image.Image), 'should be a Pil image'
        clip = [torch.from_numpy(frame) for frame in clip]  # T x C x H x W
        return torch.stack(clip, dim=1)  # C x T x H x W


class ToZeroOne:
    def __call__(self, clip):
        """
        :param clip: C x T x H x W tensor
        :return: C x T x H x W float32 tensor between 0 and 1
        """
        assert isinstance(clip, torch.Tensor)
        return clip.float() / 255.0

