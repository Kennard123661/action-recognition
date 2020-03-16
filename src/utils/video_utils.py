import torchvision.transforms._transforms_video as transforms
import torch
from PIL import Image


class ToTensor:
    def __call__(self, clip):
        assert isinstance(clip[0], Image.Image), 'should be a Pil image'
        clip = [torch.from_numpy(frame) for frame in clip]  # T x C x H x W
        return torch.stack(clip, dim=0)
