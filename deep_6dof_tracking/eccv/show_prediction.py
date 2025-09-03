import argparse

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.modelrenderer import InitOpenGL, ModelRenderer

import cv2
import os
import time
import pandas as pd
import numpy as np

from deep_6dof_tracking.data.sequence_loader import SequenceLoader
from deep_6dof_tracking.data.utils import image_blend
from deep_6dof_tracking.eccv.eval_functions import eval_pose_diff
from deep_6dof_tracking.utils.transform import Transform

ESCAPE_KEY = 1048603
SPACE_KEY = 1048608

GREEN_HUE = 120/2
RED_HUE = 230/2
BLUE_HUE = 0
LIGH_BLUE_HUE = 20
PURPLE_HUE = 300/2


def image_blend_gray(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    :param foreground:
    :param background:
    :return:
    """
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, :] == 0
        mask = np.all(mask, axis=2)[:, :, np.newaxis]
    return background[:, :, np.newaxis] * mask + foreground

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show and load sequence prediction')

    parser.add_argument('--sequence', help="sequence path", action="store")
    parser.add_argument('--predictions', help="predictions path", action="store")
    parser.add_argument('--model', help="model path", action="store")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('--show_gt', help="show the ground truth instead of prediction", action="store_true")
    parser.add_argument('--save_frames', help="Save frames path", action="store", default="")
    parser.add_argument('--brightness', help="Amount of brightness to remove", action="store", default=100, type=int)
    parser.add_argument('--color', help="R = red, G = green, B = blue, N = normal", action="store", default="N")
    parser.add_argument('--show', help="show image", action="store_true")

    # TODO
    parser.add_argument('--repair_gt', help="Change position for every frames", action="store_true")


    arguments = parser.parse_args()

    # Populate important data from config file

    PREDICTION_PATH = arguments.predictions
    SEQUENCE_PATH = arguments.sequence
    MODEL_GEO_PATH = os.path.join(arguments.model, "geometry.ply")
    MODEL_AO_PATH = os.path.join(arguments.model, "ao.ply")
    SHADER_PATH = arguments.shader
    SHOW_GT = arguments.show_gt
    SAVE_FRAME_PATH = arguments.save_frames
    BRIGHTNESS = arguments.brightness
    COLOR = arguments.color
    SHOW = arguments.show

    dataset = SequenceLoader(SEQUENCE_PATH)
    small_camera = dataset.camera.copy()
    small_camera.set_ratio(1.5)
    camera_size = (small_camera.width, small_camera.height)
    window = InitOpenGL(*camera_size)
    vpRender = ModelRenderer(MODEL_GEO_PATH, SHADER_PATH, small_camera, window, camera_size)
    vpRender.load_ambiant_occlusion_map(MODEL_AO_PATH)

    prediction_path = os.path.join(PREDICTION_PATH, "prediction_pose.csv")
    ground_truth_pose = os.path.join(PREDICTION_PATH, "ground_truth_pose.csv")
    predictions = pd.read_csv(prediction_path)
    gt = pd.read_csv(ground_truth_pose)

    pose_diff_t, pose_diff_r = eval_pose_diff(ground_truth_pose, prediction_path)
    error_frame = np.zeros((camera_size[1], camera_size[0], 3), np.uint8)
    counter = 0

    if SAVE_FRAME_PATH != "":
        if not os.path.exists(SAVE_FRAME_PATH):
            os.mkdir(SAVE_FRAME_PATH)

    hue= -1
    if COLOR == "R" or COLOR == "W":
        hue = RED_HUE
    elif COLOR == "B":
        hue = BLUE_HUE
    elif COLOR == "G":
        hue = GREEN_HUE
    elif COLOR == "LB":
        hue = LIGH_BLUE_HUE
    elif COLOR == "P":
        hue = PURPLE_HUE

    print("Sequence length: {}".format(len(dataset.data_pose)))
    for i, (frame, pose) in enumerate(dataset.data_pose[1:]):
        rgb, depth = frame.get_rgb_depth(SEQUENCE_PATH)
        rgb = cv2.resize(rgb, camera_size)
        depth = cv2.resize(depth, camera_size)

        if not SHOW_GT:
            params = predictions.iloc[i]
            if len(params) == 6:
                pose = Transform.from_parameters(*predictions.iloc[i])
            else:
                matrix = params.as_matrix().reshape((4, 4))
                pose = Transform.from_matrix(matrix)

        rgb_render, depth_render = vpRender.render_image(pose)

        render_frame = rgb_render.copy()
        if hue != -1:
            hsv = cv2.cvtColor(render_frame, cv2.COLOR_RGB2HSV).astype(int)
            hsv[:, :, 0] = hue
            hsv[:, :, 1] += 150
            hsv[:, :, 2] += 100
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            render_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            render_frame[rgb_render == 0] = 0

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray[:, :] = (np.clip(gray.astype(int) - BRIGHTNESS, 0, 255)).astype(np.uint8)
        blend = image_blend_gray(render_frame[:, :, ::-1], gray)

        if SAVE_FRAME_PATH != "":
            cv2.imwrite(os.path.join(SAVE_FRAME_PATH, '{}.png').format(i), blend[:, :, ::-1])
            #cv2.imwrite(os.path.join(SAVE_FRAME_PATH, '{}d.png').format(i), depth.astype(np.uint16))

        if SHOW:
            cv2.imshow("debug", blend[:, :, ::-1])
            cv2.waitKey(1)


