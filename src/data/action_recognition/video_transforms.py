import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image


INTERPOLATION = {
    'bilinear': {
        'cv2': cv2.INTER_LINEAR,
        'pil': Image.BILINEAR
    }
}



def _crop_pil_image(img, left, top, width, height):
    assert isinstance(img, Image.Image), 'image is not a PIL Image'
    return img.crop((left, top, left + width, top + height))


class VideoCenterCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, clip):
        start_frame = clip[0]

        if isinstance(start_frame, Image.Image):
            image_width, image_height = start_frame.size
            crop_height, crop_width = self.crop_size, self.crop_size
            crop_top = int(round((image_height - crop_height) / 2.))
            crop_left = int(round((image_width - crop_width) / 2.))

            cropped_clip = [
                _crop_pil_image(frame, crop_left, crop_top, self.crop_size, self.crop_size) for frame in clip
            ]
        else:
            raise NotImplementedError('no such implementation')
        return cropped_clip


class VideoAspectPreservingResize:
    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int)
        assert interpolation in ['bilinear', 'bicubic', 'nearest neighbors']
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        start_frame = clip[0]
        if isinstance(start_frame, np.ndarray):
            height, width = start_frame.shape[:2]
        elif isinstance(start_frame, Image.Image):
            height, width = start_frame.size
        else:
            raise ValueError('no such frame format')

        if height > width:
            ratio = self.size / width
        else:
            ratio = self.size / height

        new_height = ratio * height
        new_width = ratio * width

        resized_clip = [cv2.resize(frame, dsize=)]


