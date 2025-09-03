from deep_6dof_tracking.utils.transform import Transform
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
from deep_6dof_tracking.utils.geodesic_grid import GeodesicGrid
from deep_6dof_tracking.eccv.eval_functions import get_pose_difference, compute_pose_diff
from tqdm import tqdm
import argparse
import shutil
import os
import math
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import block_reduce


import configparser
ESCAPE_KEY = 1048603

import matplotlib.pyplot as plt
from typing import List

import random
from deep_6dof_tracking.data.rgbd_dataset import RGBDDataset
def color_blend(rgb1, depth1, rgb2, depth2):
    mask = np.all(rgb1 == 0, axis=2)
    mask_d = depth1 == 0
    mask_d = cv2.erode(mask_d.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    depth1[mask_d] = 0
    rgb1[mask, :] = 0
    mask = mask.astype(np.uint8)
    new_depth = depth2 * mask_d + depth1
    new_color = rgb2 * mask[:, :, np.newaxis] + rgb1
    return new_color.astype(np.uint8), new_depth
class Background(object):
    def __init__(self, path, max_offset_proba=0.8, newbg=False, newbg2=False, newbg3=False, image_size=(640, 480), set_index=None):
        self.background = RGBDDataset(path)
        self.max_offset_proba = max_offset_proba
        self.newbg = newbg
        self.newbg2 = newbg2
        self.newbg3 = newbg3
        
        self.color_background, self.depth_background = self.background.load_random_image_full_res(image_size[::-1], index=set_index)
        self.color_background = self.color_background[64:-64,48:-48,:]
        self.depth_background = self.depth_background[64:-64,48:-48]
        self.color_background = cv2.resize(self.color_background, image_size, interpolation=cv2.INTER_LINEAR)
        self.depth_background = cv2.resize(self.depth_background, image_size, interpolation=cv2.INTER_NEAREST)
        self.depth_background = self.depth_background[:image_size[1], :image_size[0]]
        self.depth_background = self.depth_background.astype(np.int32)
        # self.color_background = self.color_background[:image_size[1], :image_size[0]]

    def __call__(self, data):
        rgbA, depthA = data

        depth_background = self.depth_background // 4

        rgbA, depthA = color_blend(rgbA, depthA, self.color_background, depth_background)

        return rgbA, depthA
    
def dilate_depth(depth_image, kernel_size=5):
    """
    Fix for dilating a depth image, propagating edge values outward and ignoring zeros.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Create a mask for non-zero pixels
    mask = (depth_image > 0).astype(np.uint8)
    
    # Dilate the mask to expand the object's region
    dilated_mask = cv2.dilate(mask, kernel)
    
    # Dilate the depth image
    dilated_depth = cv2.dilate(depth_image, kernel)
    
    # Combine dilated depth with the dilated mask
    result = np.where(dilated_mask, dilated_depth, 0)
    return result

def erode_depth(depth_image, kernel_size=5):
    """
    Erodes a depth image, shrinking the object while ignoring the background.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Erode the image
    eroded = cv2.erode(depth_image, kernel)

    # Keep background as 0
    result = np.where(depth_image > 0, eroded, 0)
    return result

from scipy.ndimage import median_filter

def dilate_depth_median(depth_image, kernel_size=3):
    """
    Applies a median filter to a depth image to propagate edge values while smoothing.
    Ignores zeros (background) and retains valid depth values.
    """
    # Mask to ignore zeros (background)
    mask = (depth_image > 0).astype(np.uint8)
    
    # Apply median filter
    median_filtered = median_filter(depth_image, size=kernel_size)
    
    # Ensure background remains zero
    result = np.where(mask, median_filtered, 0)
    return result

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
def interpolate_pose(T1, T2, alpha):
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    
    # Interpolate rotation using spherical linear interpolation (slerp)
    # r1, r2 = R.from_matrix(R1), R.from_matrix(R2)
    slerp = Slerp([0, 1], R.from_matrix([R1, R2]))  # Create slerp object
    R_interp = slerp([alpha]).as_matrix()
    
    # Interpolate translation linearly
    t_interp = (1 - alpha) * t1 + alpha * t2
    
    # Combine into a single pose matrix
    T_interpolated = np.eye(4)
    T_interpolated[:3, :3] = R_interp
    T_interpolated[:3, 3] = t_interp
    return T_interpolated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('-c', '--camera', help="camera json path", action="store",
                        default="../data/sensors/camera_parameter_files/Helios2.json")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="model file path", action="store", default="./model_config.yaml")

    parser.add_argument('-y', '--saveformat', help="save format", action="store", default="numpy")
    parser.add_argument('-e', '--resolution', help="image pixel size", action="store", default=150, type=int)
    parser.add_argument('--boundingbox', help="Crop bounding box width in %% of the object max width",
                        action="store", default=10, type=int)

    parser.add_argument('-p', '--preload', help="Load any data saved in output directory", action="store_true")
    parser.add_argument('-v', '--show', help="show image while generating", action="store_true")
    parser.add_argument('-d', '--debug', help="show debug screen while generating", action="store_true")
    parser.add_argument('--specular', help="Will add random specularity to the training set", action="store_true")
    parser.add_argument('--depthonly', help="Only generate depth data", action="store_true")
    parser.add_argument('--pose', help="pose to generate", action="store", default="cam1")
    parser.add_argument('-b', '--background', help="background path", action="store", default="E:/enattendant")
    parser.add_argument('--bindex', help="set background index", action="store", default=None)
    parser.add_argument('-s', '--samples', help="Number of samples to generate",
                        action="store", default=532, type=int)

    parser.add_argument('--config', help="config file path", action="store", default=None)

    arguments = parser.parse_args()

    IMAGE_SIZE = (arguments.resolution, arguments.resolution)
    PRELOAD = arguments.preload
    SPECULAR = arguments.specular
    SHADER_PATH = arguments.shader
    CAMERA_PATH = arguments.camera
    BOUNDING_BOX = arguments.boundingbox
    DEPTHONLY = arguments.depthonly

    if arguments.config is None:
        SAVE_TYPE = arguments.saveformat
        SHOW = arguments.show
        DEBUG = arguments.debug
        OUTPUT_PATH = arguments.output
        MODEL = arguments.model
        # MODELS = yaml_load(arguments.model)["models"]
        POSE = arguments.pose
        bg_path = arguments.background
        bindex = arguments.bindex
        n_samples = arguments.samples

    else:
        config = configparser.ConfigParser()
        config.read(arguments.config)
        SAVE_TYPE = config['DEFAULT']['saveformat']
        SHOW = config['DEFAULT'].getboolean('show')
        DEBUG = config['DEFAULT'].getboolean('debug')
        OUTPUT_PATH = config['DEFAULT']['output']
        MODEL = config['DEFAULT']['model']
        POSE = config['DEFAULT']['pose']
        bg_path = config['DEFAULT']['background']
        bindex = config['DEFAULT']['bindex']
        try:
            bindex = int(bindex)
        except:
            bindex = None
        n_samples = int(config['DEFAULT']['samples'])

    # if SHOW:
    #     import cv2
    import cv2
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    shutil.copy(MODEL, os.path.join(OUTPUT_PATH, "model.ply"))

    # Write important misc data to file
    metadata = {}
    metadata["image_size"] = str(IMAGE_SIZE[0])
    metadata["save_type"] = SAVE_TYPE
    metadata["object_width"] = {}
    metadata["bounding_box"] = BOUNDING_BOX

    current_dir = os.path.dirname(os.path.realpath(__file__))
    camera = Camera.load_from_json(os.path.join(current_dir, CAMERA_PATH))
    IMAGE_SIZE = (camera.width, camera.height)

    dataset = DeepTrackLoaderBase(OUTPUT_PATH, read_data=PRELOAD)
    dataset.set_save_type(SAVE_TYPE)
    dataset.camera = camera
    window_size = (camera.width, camera.height)
    preload_count = 0
    print("Compute Mean bounding box")

    pose0 = np.loadtxt(os.path.join(current_dir, 'neurobot_initial_poses/festo_cam1/pose0.txt'))
    pose_finale = np.loadtxt(os.path.join(current_dir, 'neurobot_initial_poses/festo_cam1/pose_finale.txt'))
    if POSE == 'cam2':
        pose0 = np.loadtxt(os.path.join(current_dir, 'neurobot_initial_poses/festo_cam2/pose0.txt'))
        pose_finale = np.loadtxt(os.path.join(current_dir, 'neurobot_initial_poses/festo_cam2/pose_finale.txt'))

    inversion_matrix = np.array([
        [1,  0,  0,  0],  # Invert x-axis
        [ 0, 1,  0,  0],  # Invert y-axis
        [ 0,  0,  1,  0],  # Keep z-axis
        [ 0,  0,  0,  1]   # Homogeneous coordinate
    ])
    pose0 = inversion_matrix @ pose0
    pose_finale = inversion_matrix @ pose_finale

    geometry_path = MODEL
    model_3d = PlyParser(geometry_path).get_vertex()
    object_max_width = maximum_width(model_3d) * 1000
    with_add = BOUNDING_BOX / 100 * object_max_width
    width = int(object_max_width + with_add)
    
    OBJECT_WIDTH = width
    #OBJECT_WIDTH = 220
    metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    print("Mean object width : {}".format(OBJECT_WIDTH))
    # Iterate over all models from config files


    samples : List[Transform] = []
    # samples.append(Transform.from_matrix(pose0))
    # samples.append(Transform.from_matrix(pose_finale))
    alphas = np.linspace(0, 1, n_samples)
    # alphas = np.linspace(0, 1, 50)
    for a in alphas:
        samples.append(Transform.from_matrix(interpolate_pose(pose0, pose_finale, a)))
    samples_np = np.array([s.matrix.flatten() for s in samples])
    np.save(os.path.join(OUTPUT_PATH, 'poses.npy'), samples_np)

    # bg_path = 'E:/enattendant' # on avait parlé de tjrs prendre le meme bg
    background = Background(bg_path, image_size=(camera.width, camera.height), set_index=bindex)


    geometry_path = MODEL
    # ao_path = os.path.join(model["path"], "ao.ply")
    
    shader_path = os.path.join(current_dir, SHADER_PATH)
    vpRender = ModelRenderer2(geometry_path, shader_path, dataset.camera, [window_size, IMAGE_SIZE], object_max_width=OBJECT_WIDTH)
    # if os.path.exists(ao_path):
    #     vpRender.load_ambiant_occlusion_map(ao_path)
    for i, s in enumerate(samples):
        # Apply mirror transform along x-axis.
        # Weird but required to fit in the BundleSDF reference frame
        # Still need to save the ORIGINAL pose
        s_mat = s.matrix
        R_x = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, 1, 0],
            [0,  0,  0, 1]
        ])
        new_mat = R_x @ s_mat
        s_render = Transform.from_matrix(new_mat)

        bb = compute_2Dboundingbox(s_render, dataset.camera, OBJECT_WIDTH, scale=(1000, 1000, -1000))
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        # vpRender.setup_camera(camera, left, right, bottom, top)
        vpRender.setup_camera(camera, 0, camera.width, 0, camera.height)
        rgbA, depthA = vpRender.render_image(s_render, fbo_index=1)
        maskA = np.zeros_like(rgbA)
        maskA[rgbA!=0] = 255

        rgbA, depthA = background((rgbA.copy(), depthA.copy()))


        index = dataset.add_pose(rgbA, depthA, s, maskA, depthonly=DEPTHONLY)

        if i % 500 == 0:
            dataset.dump_images_on_disk()
        if i % 5000 == 0:
            dataset.save_json_files(metadata)

        if DEBUG:
            show_frames(rgbA, depthA, np.zeros_like(rgbA), np.zeros_like(depthA))
        if SHOW:
            heatmap = cv2.applyColorMap(cv2.convertScaleAbs(depthA, alpha=0.05), cv2.COLORMAP_INFERNO)
            cv2.imshow("test", np.concatenate((rgbA[:, :, ::-1], heatmap), axis=1))
            k = cv2.waitKey(1)
            if k == ESCAPE_KEY:
                break

    

    dataset.dump_images_on_disk()
    dataset.save_json_files(metadata)
