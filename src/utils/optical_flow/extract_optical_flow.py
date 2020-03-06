import argparse
import os
import sys
import torch
import numpy as np
from multiprocessing import Pool

N_GPUS = 1
if __name__ == '__main__':
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(BASE_DIR)
from config import ROOT_DIR
DENSEFLOW_DIR = os.path.join(ROOT_DIR, 'third_party', 'dense_flow', 'build')
EXTRACT_OPTICAL_FLOW_PRGM = os.path.join(DENSEFLOW_DIR, 'extract_gpu')
N_FILES = 0

import data.activitynet as anet
import data.kinetics400 as kinetics

# def dump_frames(vid_path):
#     import cv2
#     video = cv2.VideoCapture(vid_path)
#     vid_name = vid_path.split('/')[-1].split('.')[0]
#     out_full_path = os.path.join(out_path, vid_name)
#
#     fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#     try:
#         os.mkdir(out_full_path)
#     except OSError:
#         pass
#     file_list = []
#     for i in xrange(fcount):
#         ret, frame = video.read()
#         assert ret
#         cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
#         access_path = '{}/{:06d}.jpg'.format(vid_name, i)
#         file_list.append(access_path)
#     print '{} done'.format(vid_name)
#     sys.stdout.flush()
#     return file_list
#
#


def run_optical_flow(video_file, out_dir, pid=0):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    video_file = os.path.abspath(video_file)
    assert os.path.exists(video_file)
    device = pid % N_GPUS
    image_prefix = os.path.abspath(os.path.join(out_dir, 'image'))
    flowx_prefix = os.path.abspath(os.path.join(out_dir, 'flowx'))
    flowy_prefix = os.path.abspath(os.path.join(out_dir, 'flowy'))

    cmd = EXTRACT_OPTICAL_FLOW_PRGM + ' -f={0} -x={1} -y={2} -i={3} -b=20 -t=1 -d={4} -s=1 -o=dir -w=340 -h=256'\
        .format(video_file, flowx_prefix, flowy_prefix, image_prefix, device)
    os.system(cmd)
    print('{0}/{1} {2} done'.format(pid, N_FILES, video_file))
    sys.stdout.flush()
    return True


# def _generate_warp_optical_flow(vid_item, dev_id=0):
#     raise NotImplementedError
    # vid_path = vid_item[0]
    # vid_id = vid_item[1]
    # vid_name = vid_path.split('/')[-1].split('.')[0]
    # out_full_path = os.path.join(out_path, vid_name)
    # try:
    #     os.mkdir(out_full_path)
    # except OSError:
    #     pass
    #
    # current = current_process()
    # dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    # flow_x_path = '{}/flow_x'.format(out_full_path)
    # flow_y_path = '{}/flow_y'.format(out_full_path)
    #
    # cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
    #     vid_path, flow_x_path, flow_y_path, dev_id, out_format)
    #
    # os.system(cmd)
    # print('INFO: warp on {} {} done'.format(vid_id, vid_name))
    # sys.stdout.flush()
    # return True


def _parse_args():
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--data", default='activitynet', choices=['activitynet', 'kinetics'])
    parser.add_argument('--n_workers', default=40, type=int)
    parser.add_argument('--flow_type', default='tvl1', type=str, choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--n_gpu", type=int, default=torch.cuda.device_count(), help='number of GPU')
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.data == 'activitynet':
        video_dir = anet.VIDEO_DIR
        extracted_dir = anet.EXTRACTED_DIR
    else:
        raise NotImplementedError
    extracted_dir = os.path.join(extracted_dir, 'opencv-gpu')
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)

    global N_GPUS
    N_GPUS = args.n_gpu

    videos = os.listdir(video_dir)
    processed_videos = os.listdir(extracted_dir)

    global N_FILES
    videos = np.sort(np.setdiff1d(videos, processed_videos)).reshape(-1)
    N_FILES = len(videos)
    video_files = [os.path.join(video_dir, video) for video in videos]

    out_dirs = [os.path.join(extracted_dir, video) for video in videos]
    pool = Pool(args.n_workers)
    pids = list(range(len(video_files)))

    if args.flow_type == 'tvl1':
        # print(zip(video_files, out_dirs, pids))
        pool.starmap(run_optical_flow, zip(video_files, out_dirs, pids))
    else:
        raise NotImplementedError
    # elif args.flow_type == 'warp_tvl1':
    # pool.map(_generate_warp_optical_flow, zip(vid_list, xrange(len(vid_list))))


if __name__ == '__main__':
    main()
