import argparse
import configparser
import json

from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from pytorch_toolbox.transformations.compose import Compose
from deep_6dof_tracking.data.data_augmentation import Occluder, HSVNoise, Background, GaussianNoise, \
    GaussianBlur, OffsetDepth, ToTensor, Transpose, DepthDownsample
import os
import sys
from tqdm import tqdm
import numpy as np
from torch.utils import data

import cv2


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute mean')
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE")
    parser.add_argument('-b', '--background', help="Background path", metavar="FILE")
    parser.add_argument('-r', '--occluder', help="Occluder path", metavar="FILE")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1, type=int)
    parser.add_argument('--newbg', help="use new and 'improved' background", action="store_true")
    parser.add_argument('--newbg2', help="use new and 'improved' background (2)", action="store_true")
    parser.add_argument('--newbg3', help="use new and 'improved' background (3)", action="store_true")
    parser.add_argument('--bb3d', help="crop the observed data with a 3D bounding box", action="store_true")
    parser.add_argument('--bb3d_rgb', help="the pixels cropped by the 3D BB will be put to 0 in the rgb image", action="store_true")
    parser.add_argument('--bb_padding', help="padding for the 3D bounding box", action="store", default=0.5, type=float)
    parser.add_argument('--config', help="config file", metavar="FILE")

    arguments = parser.parse_args()

    if arguments.config is not None:
        config = configparser.ConfigParser()
        config.read(arguments.config)
        occluder_path = config['DEFAULT']['occluder']
        background_path = config['DEFAULT']['background']
        data_path = config['DEFAULT']['dataset']
        newbg3 = config['DEFAULT'].getboolean('newbg3')
        bb3d = config['DEFAULT'].getboolean('bb3d')
    else:
        occluder_path = arguments.occluder
        background_path = arguments.background
        data_path = arguments.dataset
        newbg3 = arguments.newbg3
        bb3d = arguments.bb3d

    data_path = os.path.expandvars(data_path)
    occluder_path = os.path.expandvars(occluder_path)
    background_path = os.path.expandvars(background_path)
    newbg = arguments.newbg
    newbg2 = arguments.newbg2
    bb3d_rgb = arguments.bb3d_rgb
    bb_padding = arguments.bb_padding

    number_of_core = arguments.ncore
    if number_of_core == -1:
        number_of_core = os.cpu_count()
    batch_size = 32

    with open(os.path.join(data_path, "train", "viewpoints.json")) as data_file:
        metadata = json.load(data_file)["metaData"]

    transformations_pre = [Compose([Occluder(occluder_path, 0.75)])]

    transformations_post = [Compose([HSVNoise(0.07, 0.05, 0.1),
                                     Background(background_path, newbg=newbg, newbg2=newbg2, newbg3=newbg3),
                                     OffsetDepth(),
                                     GaussianNoise(2, 5),
                                     GaussianBlur(6),
                                     DepthDownsample(0.7),
                                     Transpose(),
                                     ToTensor()])]
    
    if bb3d:
        from deep_6dof_tracking.data.data_augmentation import BoundingBox3D
        transformations_post = [Compose([HSVNoise(0.07, 0.05, 0.1),
                                     Background(background_path, newbg=newbg, newbg2=newbg2, newbg3=newbg3),
                                     OffsetDepth(),
                                     GaussianNoise(2, 5),
                                     GaussianBlur(6),
                                     DepthDownsample(0.7),
                                     BoundingBox3D(object_max_width=float(metadata['bounding_box_width']), original_padding=0.15, padding=bb_padding, crop_rgb=bb3d_rgb),
                                     Transpose(),
                                     ToTensor()])]

    train_dataset = DeepTrackLoader(os.path.join(data_path, "train"), transformations_pre, transformations_post)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=False,
                                   drop_last=True,
                                   )

    n = 20000
    channel_means = np.zeros(8)
    total = 0
    for i, (data, target) in tqdm(enumerate(train_loader), total=int(n / batch_size)):
        bufferA, bufferB = data
        bufferA_numpy = bufferA.cpu().numpy()
        bufferB_numpy = bufferB.cpu().numpy()
        buffer_numpy = np.concatenate((bufferA_numpy, bufferB_numpy), axis=1)
        channel_means += np.mean(buffer_numpy, axis=(0, 2, 3))
        total += 1
        if i * batch_size >= n:
            break
    channel_means = channel_means / total

    channel_std = np.zeros(8)
    total = 0
    for i, (data, target) in tqdm(enumerate(train_loader), total=int(n / batch_size)):
        bufferA, bufferB = data
        bufferA_numpy = bufferA.cpu().numpy()
        bufferB_numpy = bufferB.cpu().numpy()
        buffer_numpy = np.concatenate((bufferA_numpy, bufferB_numpy), axis=1)
        image_means = np.mean(buffer_numpy, axis=(0, 2, 3))
        channel_std += np.square(image_means - channel_means)
        total += 1
        if i * batch_size >= n:
            break
    channel_std = np.sqrt(channel_std / total)

    print(channel_means)
    print(channel_std)
    np.save(os.path.join(data_path, "mean.npy"), channel_means)
    np.save(os.path.join(data_path, "std.npy"), channel_std)
