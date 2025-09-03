"""
    use a pose detector (aruco, checkboard) and compute the pose on the whole dataset
"""
import argparse
import sys
from pytorch_toolbox.io import yaml_load
from tqdm import tqdm

from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.utils import image_blend
from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
import cv2
import os
import numpy as np


if __name__ == '__main__':
    """
    Simple tool to remove frames
    """
    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('--dataset1', help="First dataset to merge", action="store")
    parser.add_argument('--dataset2', help="Second dataset to merge", action="store")
    parser.add_argument('--output', help="Output dataset", action="store")
    parser.add_argument('--max_count', help="Maximum images in output dataset", action="store", default=200000, type=int)

    arguments = parser.parse_args()

    dataset_path1 = arguments.dataset1
    dataset_path2 = arguments.dataset2
    output_path = arguments.output
    maximum_count = arguments.max_count

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    dataset1 = DeepTrackLoader(dataset_path1)
    dataset2 = DeepTrackLoader(dataset_path2)
    dataset_out = DeepTrackLoader(output_path)

    metadata = dataset1.metadata
    dataset_out.set_save_type(dataset1.metadata["save_type"])
    dataset_out.camera = dataset1.camera
    count = 0

    print("Load dataset 1")
    for i, (frame, pose) in tqdm(enumerate(dataset1.data_pose), total=len(dataset1.data_pose)):
        rgb, depth = frame.get_rgb_depth(dataset1.root)
        out_id = dataset_out.add_pose(rgb, depth, pose)
        for j in range(dataset1.pair_size(frame.id)):
            rgb_pair, depth_pair, pose_pair = dataset1.load_pair(frame.id, j)
            dataset_out.add_pair(rgb_pair, depth_pair, pose_pair, out_id)
        count += 1

        if count % 5000 == 0:
            dataset_out.dump_images_on_disk()
            dataset_out.save_json_files(metadata)

        if count > maximum_count:
            print("[warn] reached a maximum of {}.".format(count))
            break

    print("Load dataset 2")
    for i, (frame, pose) in tqdm(enumerate(dataset2.data_pose), total=len(dataset2.data_pose)):
        rgb, depth = frame.get_rgb_depth(dataset2.root)
        out_id = dataset_out.add_pose(rgb, depth, pose)
        for j in range(dataset2.pair_size(frame.id)):
            rgb_pair, depth_pair, pose_pair = dataset2.load_pair(frame.id, j)
            dataset_out.add_pair(rgb_pair, depth_pair, pose_pair, out_id)
        count += 1

        if count % 5000 == 0:
            dataset_out.dump_images_on_disk()
            dataset_out.save_json_files(metadata)

        if count > maximum_count:
            print("[warn] reached a maximum of {}.".format(count))
            break

    dataset_out.dump_images_on_disk()
    dataset_out.save_json_files(metadata)
