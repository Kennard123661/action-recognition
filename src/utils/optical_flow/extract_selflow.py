from __future__ import division, print_function, absolute_import
import torch
import tensorflow as tf
import numpy as np
import os
import argparse

from nets.optical_flow.selflow import flow_resize, pyramid_processing
import nets.optical_flow.selflow as selflow


def mvn(img):
    # minus mean color and divided by standard variance
    mean, var = tf.nn.moments(img, axes=[0, 1], keep_dims=True)
    img = (img - mean) / tf.sqrt(var + 1e-12)
    return img


def write_flow(save_file, flow):
    raise NotImplementedError


class BasicDataset(object):
    def __init__(self, crop_h=320, crop_w=896, batch_size=4, data_list_file='path_to_your_data_list_file',
                 img_dir='path_to_your_image_directory', fake_flow_occ_dir='path_to_your_fake_flow_occlusion_directory',
                 is_normalize_img=True):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.data_list = np.loadtxt(data_list_file, dtype=bytes).astype(np.str)
        self.data_num = self.data_list.shape[0]
        self.fake_flow_occ_dir = fake_flow_occ_dir
        self.is_normalize_img = is_normalize_img

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
        img0_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[1]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[2]])

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
        dataset = dataset.batch(1)
        dataset = dataset.repeat()
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


def _extract_optical_flow(dataset, ckpt_file, batch_size=8, n_workers=4, n_gpus=1, cpu_device='/cpu:0'):
    batch_size = int(batch_size)
    n_workers = int(n_workers)
    n_gpus = int(n_gpus)
    shared_device = '/gpu:0' if n_gpus == 1 else cpu_device


class SelFlowInference:
    def __init__(self, pretrained_ckpt, batch_size=8, n_workers=4, n_gpus=1, cpu_device='/cpu:0'):
        self.pretrained_file = os.path.join(SELFLOW_PRETRAINED_DIR, pretrained_ckpt)
        assert os.path.exists(self.pretrained_file), 'checkpoint file {} does not exist.'
        self.batch_size = int(batch_size)
        self.n_workers = int(n_workers)
        self.n_gpus = int(n_gpus)
        self.shared_device = '/gpu:0' if self.n_gpus == 1 else cpu_device


class SelFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5,
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False,
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints',
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_distillation",
                 is_restore_model=False, restore_model='./models/KITTI/no_census_no_occlusion',
                 dataset_config={}, self_supervision_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.training_mode = training_mode
        self.is_restore_model = is_restore_model
        self.restore_model = restore_model
        self.dataset_config = dataset_config
        self.self_supervision_config = self_supervision_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        assert (np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))

        self.sample_dir = '/'.join([self.save_dir, sample_dir])
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists('/'.join([self.sample_dir, self.model_name])):
            os.makedirs(('/'.join([self.sample_dir, self.model_name])))

        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train'])))
        if not os.path.exists('/'.join([self.summary_dir, 'test'])):
            os.makedirs(('/'.join([self.summary_dir, 'test'])))

    def test(self, restore_model, save_dir, is_normalize_img=True):
        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'],
                               img_dir=self.dataset_config['img_dir'], is_normalize_img=is_normalize_img)
        save_name_list = dataset.data_list[:, -1]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
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
        saver.restore(sess, restore_model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(dataset.data_num):
            np_flow_fw, np_flow_bw, np_flow_fw_color, np_flow_bw_color = sess.run(
                [flow_fw['full_res'], flow_bw['full_res'], flow_fw_color, flow_bw_color])
            # todo: implement save function here.
            # misc.imsave('%s/flow_fw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_fw_color[0])
            # misc.imsave('%s/flow_bw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_bw_color[0])
            # write_flo('%s/flow_fw_%s.flo' % (save_dir, save_name_list[i]), np_flow_fw[0])
            # write_flo('%s/flow_bw_%s.flo' % (save_dir, save_name_list[i]), np_flow_bw[0])
            print('Finish %d/%d' % (i + 1, dataset.data_num))


def _parse_args():
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument('--ckpt', default='sintel', type=str, choices=['kitti', 'sintel'])
    parser.add_argument("--data", default='activitynet', choices=['activitynet', 'kinetics'])
    parser.add_argument('--n_workers', default=8, type=int)
    parser.add_argument("--n_gpu", type=int, default=torch.cuda.device_count(), help='number of GPU')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    return parser.parse_args()


def main():
    raise NotImplementedError('deprecated in favor of Nvidia Toolkit')
    args = _parse_args()
    if args.ckpt == 'sintel':
        ckpt_file = selflow.SINTEL_PRETRAINED_DIR
    else:
        ckpt_file = selflow.KITTI_PRETRAINED_DIR
    ckpt_file = os.path.join(ckpt_file, 'supervise_finetune')

    _extract_optical_flow(ckpt_file=ckpt_file, batch_size=args.batch_size, n_workers=args.n_workers,
                          n_gpus=args.n_gpus)



if __name__ == '__main__':
    main()


