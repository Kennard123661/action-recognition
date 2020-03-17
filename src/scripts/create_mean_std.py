import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import data.breakfast as breakfast
from torchvision.transforms import ToTensor


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='breakfast')
    return argparser.parse_args()


def compute_mean_std(video_dirs):
    """ reference from https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time """
    n_frames = 0
    mean = np.zeros(shape=3).astype(np.float128)
    variance = np.zeros(shape=3).astype(np.float128)

    to_tensor = ToTensor()
    for video_dir in tqdm(video_dirs):
        frames = sorted(os.listdir(video_dir))
        for frame in frames:
            frame_file = os.path.join(video_dir, frame)
            image = Image.open(frame_file)
            image = to_tensor(image)
            channels = torch.mean(torch.mean(image, dim=2), dim=1).numpy()

            n_frames += 1
            prev_mean = np.copy(mean)
            mean = mean + (channels - mean) / n_frames
            variance = variance + (channels - mean) * (channels - prev_mean)
    std = np.sqrt(variance / n_frames)
    return mean, std


def main():
    args = parse_args()
    if args.dataset == 'breakfast':
        train_segments, _, _ = breakfast.get_data(split='train')
    else:
        raise ValueError('no such dataset')

    video_names = []
    for segment in train_segments:
        video_name = segment['video-name']
        video_names.append(video_name)

    video_names = np.unique(video_names).reshape(-1)
    video_dirs = [os.path.join(breakfast.EXTRACTED_IMAGES_DIR, video) for video in video_names]
    mean, std = compute_mean_std(video_dirs=video_dirs)
    np.save(breakfast.MEAN_FILE, mean)
    np.save(breakfast.STD_FILE, std)


if __name__ == '__main__':
    main()
