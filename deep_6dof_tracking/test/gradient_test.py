from PIL import Image
from pytorch_toolbox.probe.activation import show_activations
from torch.autograd import Variable
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.data.modelrenderer import InitOpenGL, ModelRenderer
from deep_6dof_tracking.deeptracker import DeepTracker

import sys
from tqdm import tqdm
import cv2
import numpy as np
import os

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale, \
    color_blend

ESCAPE_KEY = 27


if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "gradient_test_config.yml"
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

    camera = Camera.load_from_json(CAMERA_PATH)
    tracker = DeepTracker(camera, BACKEND, ARCHITECTURE, debug=DEBUG)
    tracker.load(MODEL_PATH, MODEL_3D_PATH_GEO, MODEL_3D_PATH_AO, SHADER_PATH)

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

    save_image = False
    # Instantiate renderer
    if save_image:
        window = InitOpenGL(camera.width, camera.height)
        renderer = ModelRenderer(MODEL_3D_PATH_GEO, SHADER_PATH, camera, window, (camera.width, camera.height))
        renderer.load_ambiant_occlusion_map(MODEL_3D_PATH_AO)
        current_rgb, current_depth = renderer.render_image(pair_pose)
        np.save("rgbA.npy", current_rgb)
        np.save("depthA.npy", current_depth)
        sys.exit()

    # background
    current_rgb = np.load("rgbA.npy")
    current_depth = np.load("depthA.npy")
    background = cv2.imread("background.png")[100:, 100:, :]
    background_d = np.array(Image.open("background_depth.png")).astype(np.uint16)[100:, 100:]
    background = cv2.resize(background, (current_rgb.shape[1], current_rgb.shape[0]))
    background_d = cv2.resize(background_d, (current_depth.shape[1], current_depth.shape[0]))

    current_rgb, current_depth = color_blend(current_rgb, current_depth, background, background_d)

    prediction_without, imgA, imgB = tracker.estimate_current_pose(object_pose,
                                                                    current_rgb,
                                                                    current_depth,
                                                                    debug=True,
                                                                    raw_prediction=True)
    activation_without_occlusion = tracker.tracker_model.load_activations()
    tracker.tracker_model.activations = []

    plt.imshow(current_rgb)
    plt.show()
    plt.imshow(current_depth)
    plt.show()

    # occluder
    #current_rgb[522:681, 965:1628, :] = 0
    #current_depth[522:681, 965:1628] = 800

    # Missing channel
    current_rgb[:, :, :] = 0
    #current_depth[:, :] = 0


    prediction_with, imgA, imgB = tracker.estimate_current_pose(object_pose,
                                                                       current_rgb,
                                                                       current_depth,
                                                                       debug=True,
                                                                       raw_prediction=True)
    activation_with_occlusion = tracker.tracker_model.load_activations()
    #predicted_pose_with = tracker.bin_to_pose(prediction_with, offset=1)
    #predicted_pose_without = tracker.bin_to_pose(prediction_without, offset=1)
    #print("Prediction results:")
    #print("Ground truth : {}".format(pair_transform.to_parameters()))
    #print("Prediction without : {}".format(predicted_pose_without))
    #print("Prediction with : {}".format(predicted_pose_with))

    for key in tqdm(activation_without_occlusion.keys()):
        a_without = activation_without_occlusion[key][0]
        a_with = activation_with_occlusion[key][0]
        #show_activations(a_with, title=key)
        # show_activations(a_without)
        show_activations(np.abs(a_with - a_without), title=key, min=0, max=3)
        plt.show()
    #l_tx = nn.KLDivLoss()(prediction_without[0], targets[0])
    #l_ty = nn.KLDivLoss()(prediction_without[1], targets[1])
    #l_tz = nn.KLDivLoss()(prediction_without[2], targets[2])

    #l_rx = nn.KLDivLoss()(prediction_without[3], targets[3])
    #l_ry = nn.KLDivLoss()(prediction_without[4], targets[4])
    #l_rz = nn.KLDivLoss()(prediction_without[5], targets[5])

    #total_loss = l_tx + l_ty + l_tz + l_rx + l_ry + l_rz
    #total_loss.backward()

    gradA = imgA.grad.data.cpu().numpy().T
    gradB = imgB.grad.data.cpu().numpy().T

    maximum = max(gradA.max(), gradB.max())
    minimum = min(gradA.min(), gradB.min())

    # Print stats
    print("Mean abs :")
    print("A : {}, B : {}".format(np.abs(gradA.mean()), np.abs(gradB.mean())))
    print("Median abs :")
    print("A : {}, B : {}".format(np.median(np.abs(gradA)), np.median(np.abs(gradB))))
    print("Maximums :")
    print("A : {}, B : {}".format(gradA.max(), gradB.max()))
    print("A = R : {}, G : {}, B : {}, D : {}".format(gradA[:, :, 0, 0].max(),
                                                      gradA[:, :, 1, 0].max(),
                                                      gradA[:, :, 2, 0].max(),
                                                      gradA[:, :, 3, 0].max()))
    print("B = R : {}, G : {}, B : {}, D : {}".format(gradB[:, :, 0, 0].max(),
                                                      gradB[:, :, 1, 0].max(),
                                                      gradB[:, :, 2, 0].max(),
                                                      gradB[:, :, 3, 0].max()))
    print("Minimums :")
    print("A : {}, B : {}".format(gradA.min(), gradB.min()))
    print("A = R : {}, G : {}, B : {}, D : {}".format(gradA[:, :, 0, 0].min(),
                                                      gradA[:, :, 1, 0].min(),
                                                      gradA[:, :, 2, 0].min(),
                                                      gradA[:, :, 3, 0].min()))
    print("B = R : {}, G : {}, B : {}, D : {}".format(gradB[:, :, 0, 0].min(),
                                                      gradB[:, :, 1, 0].min(),
                                                      gradB[:, :, 2, 0].min(),
                                                      gradB[:, :, 3, 0].min()))

    for i in range(4):
        plt.subplot("24{}".format(i + 1))
        plt.imshow(gradA[:, :, i, 0], vmin=minimum / 10, vmax=maximum / 100)
        plt.subplot("24{}".format(i + 5))
        plt.imshow(gradB[:, :, i, 0], vmin=minimum / 10, vmax=maximum / 100)
    plt.show()

    """
    cv2.imshow("Debug", screen[:, :, ::-1])
    key = cv2.waitKey(1)
    key_chr = chr(key & 255)
    if key != -1:
        print("pressed key id : {}, char : [{}]".format(key, key_chr))
    if key == ESCAPE_KEY:
        break
    """
