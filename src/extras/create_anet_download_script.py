from argparse import ArgumentParser
import glob
import json
import os

import data.activitynet as anet


def crosscheck_videos(video_path, ann_file):
    # Get existing videos
    existing_vids = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_vids):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[idx] = basename[2:]
        elif len(basename) == 11:
            existing_vids[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    # Read an get video IDs from annotation file
    with open(ann_file, "r") as fobj:
        anet_v_1_0 = json.load(fobj)
    all_vids = anet_v_1_0["database"].keys()
    non_existing_videos = []
    for vid in all_vids:
        if vid in existing_vids:
            continue
        else:
            non_existing_videos.append(vid)
    return non_existing_videos


def main():
    """ generates bash script to run activity net download """
    video_path = anet.VIDEO_DIR
    ann_file = anet.ANET13_ANNOTATION_JSON
    output_filename = anet.ANET13_DOWNLOAD_SCRIPT

    non_existing_videos = crosscheck_videos(video_path, ann_file)
    filename = os.path.join(video_path, "v_%s.mp4")
    cmd_base = "youtube-dl -f best -f mp4 "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s"' % filename

    print('INFO: download script generated at {}'.format(output_filename))
    with open(output_filename, "w") as fobj:
        for vid in non_existing_videos:
            cmd = cmd_base % (vid, vid)
            fobj.write("%s\n" % cmd)


if __name__ == "__main__":
    main()
