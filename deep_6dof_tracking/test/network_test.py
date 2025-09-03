from PIL import Image
from pytorch_toolbox.probe.activation import show_activations
from torch.autograd import Variable
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.data.modelrenderer import InitOpenGL, ModelRenderer

import sys
from tqdm import tqdm
import cv2
import numpy as np
import os

from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale, \
    color_blend

ESCAPE_KEY = 27

if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "network_test_config.yml"
    configs = yaml_load(config_path)

    # Populate important data from config file
    OUTPUT_PATH = configs["output_path"]
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    MODEL_PATH = configs["model_path"]
    MODEL_3D_PATH = configs["model_3d_path"]
    SHADER_PATH = configs["shader_path"]

    VERBOSE = configs["verbose"]
    CAMERA_PATH = configs["camera_path"]
    BACKEND = configs["backend"]
    ARCHITECTURE = configs["architecture"]
    DEBUG = configs["debug"]

    model_split_path = MODEL_PATH.split(os.sep)
    model_name = model_split_path[-1]
    model_folder = os.sep.join(model_split_path[:-1])

    MODEL_3D_PATH_GEO = os.path.join(MODEL_3D_PATH, "geometry.ply")
    MODEL_3D_PATH_AO = os.path.join(MODEL_3D_PATH, "ao.ply")
    if not os.path.exists(MODEL_3D_PATH_AO):
        MODEL_3D_PATH_AO = None

    checkpoints = ["model_best.pth.tar"]

    camera = Camera.load_from_json(CAMERA_PATH)
    tracker = DeepTrackerBatch(camera, BACKEND, ARCHITECTURE)
    tracker.load(os.path.join(MODEL_PATH, checkpoints[0]), MODEL_3D_PATH_GEO, MODEL_3D_PATH_AO, SHADER_PATH)

    object_pose = Transform.from_parameters(0, 0, -1, 0, 0, 0)
    pair_transform = Transform.from_parameters(0, 0.019, 0, 0, 0, 0)
    pair_pose = combine_view_transform(object_pose, pair_transform)

    ## hardcoded! Build bins
    n_bins = 41
    max_t_val = 0.02
    min_t_val = -max_t_val
    step = (max_t_val - min_t_val) / n_bins
    translation_bins = np.linspace(min_t_val, max_t_val - step, n_bins)

    max_r_val = math.radians(10)
    min_r_val = -max_r_val
    step = (max_r_val - min_r_val) / n_bins
    rotation_bins = np.linspace(min_r_val, max_r_val - step, n_bins)

    # Compute target
    pose_label = pair_pose.to_parameters()
    binned_labels = np.zeros(6, dtype=int)
    binned_labels[:3] = np.digitize(pose_label[:3], translation_bins)
    binned_labels[3:] = np.digitize(pose_label[3:], rotation_bins)
    targets = np.zeros((6, n_bins), dtype=np.float32)
    for i, label in enumerate(binned_labels):
        targets[i, label] = 1
    targets = Variable(torch.from_numpy(targets))
    if BACKEND == "cuda":
        targets = targets.cuda()

    tracker.setup_renderer(MODEL_3D_PATH_GEO, None, SHADER_PATH)


    # background
    current_rgb = np.load("rgbA.npy")
    current_depth = np.load("depthA.npy")
    background = cv2.imread("background.png")[100:, 100:, :]
    background_d = np.array(Image.open("background_depth.png")).astype(np.uint16)[100:, 100:]
    background = cv2.resize(background, (current_rgb.shape[1], current_rgb.shape[0]))
    background_d = cv2.resize(background_d, (current_depth.shape[1], current_depth.shape[0]))

    current_rgb, current_depth = color_blend(current_rgb, current_depth, background, background_d)

    predictionA, imgA, imgB = tracker.estimate_current_pose(object_pose,
                                                            current_rgb,
                                                            current_depth,
                                                            raw_prediction=True)
    activationA = tracker.tracker_model.load_activations()

    print(activationA)

    for key in activationA.keys():
        show_activations(np.abs(activationA[key][0]), title=key, min=0, max=1)
        plt.show()

