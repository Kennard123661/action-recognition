from __future__ import division, print_function, absolute_import
import torch
import tensorflow as tf
import numpy as np
import os
import math
import argparse
import imageio
from tqdm import tqdm

from config import ROOT_DIR
from nets.optical_flow.selflow import flow_resize, pyramid_processing
import nets.optical_flow.selflow as selflow
import data.activitynet as anet
import data.breakfast as breakfast
import data.kinetics400 as kinetics


def mvn(img):
    # minus mean color and divided by standard variance
    mean, var = tf.nn.moments(img, axes=[0, 1], keep_dims=True)
    img = (img - mean) / tf.sqrt(var + 1e-12)
    return img


def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


class BasicDataset(object):
    def __init__(self, image_dirs, out_dirs, crop_h=320, crop_w=896, batch_size=4, is_normalize_img=True):
        self.image_dirs = image_dirs
        self.out_dirs = out_dirs
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)
        self.batch_size = int(batch_size)
        self.is_normalize_img = bool(is_normalize_img)

        print('INFO: generating file queues...')
        file_queues = []
        out_fids = []
        for d, image_dir in enumerate(tqdm(self.image_dirs)):
            n_images = len(os.listdir(image_dir))
            # image_files = [os.path.join(image_dir, '{0:06d}.jpg'.format(i)) for i in range(n_images)]
            image_files = [os.path.join(image_dir, '{0:06d}.jpg'.format(i+1)) for i in range(n_images)]
            for file in image_files:
                assert os.path.exists(file), '{} does not exist'.format(file)

            out_fids += [os.path.join(out_dirs[d], '{0:06d}'.format(i)) for i in range(n_images)]
            for i in range(n_images):
                prev_i = max(0, i-1)
                next_i = min(n_images-1, i+1)
                file_queue = [image_files[prev_i], image_files[i], image_files[next_i]]
                file_queues.append(file_queue)
        self.file_queues = np.array(file_queues)
        self.out_fids = np.array(out_fids)
        self.n_data = self.out_fids.shape[0]

    # KITTI's data format for storing flow and mask
    # The first two channels are flow, the third channel is mask
    def extract_flow_and_mask(self, flow):
        # The default image type is PNG.
        optical_flow = flow[:, :, :2]
        optical_flow = (optical_flow - 32768) / 64.0
        mask = tf.cast(tf.greater(flow[:, :, 2], 0), tf.float32)
        mask = tf.expand_dims(mask, -1)
        return optical_flow, mask

    def read_and_decode(self, filename_queue):
        img0_name = filename_queue[0]
        img1_name = filename_queue[1]
        img2_name = filename_queue[2]

        img0 = tf.image.decode_png(tf.read_file(img0_name), channels=3)
        img0 = tf.cast(img0, tf.float32)
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)
        return img0, img1, img2

    # For Validation or Testing
    def preprocess_one_shot(self, filename_queue):
        img0, img1, img2 = self.read_and_decode(filename_queue)
        img0 = img0 / 255.
        img1 = img1 / 255.
        img2 = img2 / 255.

        if self.is_normalize_img:
            img0 = mvn(img0)
            img1 = mvn(img1)
            img2 = mvn(img2)
        return img0, img1, img2

    def create_one_shot_iterator(self, data_list, num_parallel_calls=4):
        """ For Validation or Testing
            Generate image and flow one_by_one without cropping, image and flow size may change every iteration
        """
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_one_shot, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator


def _flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.
    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    """
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(tf.to_float(max_flow), 1.)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = tf.atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask


def _extract_optical_flow(image_dirs, out_dirs, model_ckpt, batch_size=4, n_workers=4, n_gpus=1, cpu_device='/cpu:0'):
    batch_size = int(batch_size)
    n_workers = int(n_workers)
    n_gpus = int(n_gpus)
    shared_device = '/gpu:0' if n_gpus == 1 else cpu_device

    dataset = BasicDataset(image_dirs=image_dirs, out_dirs=out_dirs, batch_size=batch_size)
    out_fids = dataset.out_fids

    iterator = dataset.create_one_shot_iterator(dataset.file_queues, num_parallel_calls=n_workers)
    batch_img0, batch_img1, batch_img2 = iterator.get_next()
    img_shape = tf.shape(batch_img0)
    h = img_shape[1]
    w = img_shape[2]

    new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
    new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)

    batch_img0 = tf.image.resize_images(batch_img0, [new_h, new_w], method=1, align_corners=True)
    batch_img1 = tf.image.resize_images(batch_img1, [new_h, new_w], method=1, align_corners=True)
    batch_img2 = tf.image.resize_images(batch_img2, [new_h, new_w], method=1, align_corners=True)

    flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False,
                                          is_scale=True)
    flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
    flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)

    flow_fw_color = _flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
    flow_bw_color = _flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)

    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=restore_vars)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    saver.restore(sess, model_ckpt)

    n_iterations = math.ceil(dataset.n_data / batch_size)
    for i in tqdm(range(n_iterations)):
        np_flow_fw, np_flow_bw, np_flow_fw_color, np_flow_bw_color = \
            sess.run([flow_fw['full_res'], flow_bw['full_res'], flow_fw_color, flow_bw_color])
        batch_fids = out_fids[i*batch_size:min((i+1)*batch_size, dataset.n_data)]
        for f, out_fid in enumerate(batch_fids):
            save_dir, fid = os.path.split(out_fid)
            imageio.imsave(os.path.join(save_dir, 'flow_fw_color_{}.png'.format(fid)), np_flow_fw_color[f])
            imageio.imsave(os.path.join(save_dir, 'flow_bw_color_{}.png'.format(fid)), np_flow_bw_color[f])
            write_flo(os.path.join(save_dir, 'flow_fw_{}.flo'.format(fid)), np_flow_fw[f])
            write_flo(os.path.join(save_dir, 'flow_bw_{}.flo'.format(fid)), np_flow_bw[f])


def _parse_args():
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument('--ckpt', default='sintel', type=str, choices=['kitti', 'sintel'])
    parser.add_argument("--dataset", default='breakfast', choices=['breakfast', 'activitynet', 'kinetics'])
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument("--n_gpu", type=int, default=torch.cuda.device_count(), help='number of GPU')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu', type=int, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.ckpt == 'sintel':
        ckpt_file = selflow.SINTEL_PRETRAINED_DIR
    else:
        ckpt_file = selflow.KITTI_PRETRAINED_DIR
    assert os.path.exists(ckpt_file), 'ckpt {} does not exist'.format(ckpt_file)
    ckpt_file = os.path.join(ckpt_file, 'supervise_finetune')

    if args.dataset == 'activitynet':
        extracted_images_dir = anet.EXTRACTED_IMAGES_DIR
        extracted_dir = anet.EXTRACTED_DIR
        n_frames_dir = anet.N_VIDEO_FRAMES_DIR
    elif args.dataset == 'breakfast':
        extracted_images_dir = breakfast.EXTRACTED_IMAGES_DIR
        extracted_dir = breakfast.EXTRACTED_DIR
        n_frames_dir = breakfast.N_VIDEO_FRAMES_DIR
    else:
        raise ValueError('no such dataset')

    selflow_out_dir = os.path.join(extracted_dir, 'selflow')
    if not os.path.exists(selflow_out_dir):
        os.makedirs(selflow_out_dir)

    # find videos that have not been processed
    videos = sorted(os.listdir(extracted_images_dir))
    videos = [video for i, video in enumerate(videos) if ((i % args.n_gpu) == args.gpu)]
    if ROOT_DIR == '/mnt/BigData/cs5242-project':
        args.n_gpu = 3
        if args.gpu == 2:
            args.gpu = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    video_selflow_dirs = [os.path.join(selflow_out_dir, video) for video in videos]
    n_frame_files = [os.path.join(n_frames_dir, video + '.npy') for video in videos]
    processed_videos = []
    for i, dirname in enumerate(video_selflow_dirs):
        if os.path.exists(dirname):
            n_flow_files = len(os.listdir(dirname))
            n_frames = np.load(n_frame_files[i])
            if n_flow_files != n_frames*4:
                assert n_flow_files <= n_frames*4
            else:
                processed_videos.append(videos[i])
        else:
            os.makedirs(dirname)

    # get videos that have been processed
    videos = np.setdiff1d(videos, processed_videos).reshape(-1)  # remove videos that are already processed
    video_selflow_dirs = [os.path.join(selflow_out_dir, video) for video in videos]
    video_image_dirs = [os.path.join(extracted_images_dir, video) for video in videos]

    _extract_optical_flow(video_image_dirs, video_selflow_dirs, model_ckpt=ckpt_file, batch_size=args.batch_size,
                          n_workers=args.n_workers, n_gpus=args.n_gpu)


if __name__ == '__main__':
    main()
