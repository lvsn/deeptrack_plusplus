import argparse

import math

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

ESCAPE_KEY = 1048603
NUM_PAD_1_KEY = 1114033
NUM_PAD_2_KEY = 1114034
NUM_PAD_3_KEY = 1114035
NUM_PAD_4_KEY = 1114036
NUM_PAD_5_KEY = 1114037
NUM_PAD_6_KEY = 1114038
NUM_PAD_7_KEY = 1114039
NUM_PAD_8_KEY = 1114040
NUM_PAD_9_KEY = 1114041
NUM_PLUS_KEY = 1114027
NUM_MINUS_KEY = 1114029
ARROW_LEFT_KEY = 1113937
ARROW_UP_KEY = 1113938
ARROW_RIGHT_KEY = 1113939
ARROW_DOWN_KEY = 1113940

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
    parser.add_argument('--model', help="model path", action="store")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('--show', help="just show the model", action="store_true")


    arguments = parser.parse_args()

    # Populate important data from config file

    SEQUENCE_PATH = arguments.sequence
    MODEL_GEO_PATH = os.path.join(arguments.model, "geometry.ply")
    MODEL_AO_PATH = os.path.join(arguments.model, "ao.ply")
    SHADER_PATH = arguments.shader
    show = arguments.show

    dataset = SequenceLoader(SEQUENCE_PATH)
    small_camera = dataset.camera.copy()
    small_camera.set_ratio(1.5)
    camera_size = (small_camera.width, small_camera.height)
    window = InitOpenGL(*camera_size)
    vpRender = ModelRenderer(MODEL_GEO_PATH, SHADER_PATH, small_camera, window, camera_size)
    vpRender.load_ambiant_occlusion_map(MODEL_AO_PATH)

    fixed_poses = []

    print("Sequence length: {}".format(len(dataset.data_pose)))
    hide_blend = False
    last_pose = None
    for i, (frame, pose) in enumerate(dataset.data_pose):
        rgb, depth = frame.get_rgb_depth(SEQUENCE_PATH)
        rgb = cv2.resize(rgb, camera_size)
        depth = cv2.resize(depth, camera_size)

        save = False
        while not save:
            rgb_render, depth_render = vpRender.render_image(pose)
            blend = rgb
            if not hide_blend:
                blend = image_blend(rgb_render[:, :, ::-1], rgb)

            cv2.imshow("debug", blend[:, :, ::-1])
            if show:
                cv2.waitKey(1)
                break
            key = cv2.waitKey()
            key_chr = chr(key & 255)
            if key != -1:
                print("pressed key id : {}, char : [{}]".format(key, key_chr))
            if key == ESCAPE_KEY:
                break
            elif key_chr == ' ':
                save = True
                fixed_poses.append(list(pose.matrix.flatten()))
                last_pose = pose
            elif key_chr == 'b':
                hide_blend = not hide_blend
            elif key_chr == 'l':
                if last_pose is not None:
                    pose = last_pose

            # Lock offset makes sure that we wont change the file from an already generated dataset... It is important
            # since we do not want to have a different offset for each pictures. offset file is only used to compute
            # ground truth object pose given images
            if key == NUM_PAD_1_KEY:
                pose.rotate(z=math.radians(-1))
            elif key == NUM_PAD_2_KEY:
                pose.translate(z=0.001)
            elif key == NUM_PAD_3_KEY:
                pose.rotate(x=math.radians(-1))
            elif key == NUM_PAD_4_KEY:
                pose.translate(x=-0.001)
            elif key == NUM_PAD_6_KEY:
                pose.translate(x=0.001)
            elif key == NUM_PAD_7_KEY:
                pose.rotate(z=math.radians(1))
            elif key == NUM_PAD_8_KEY:
                pose.translate(z=-0.001)
            elif key == NUM_PAD_9_KEY:
                pose.rotate(x=math.radians(1))
            elif key == ARROW_UP_KEY:
                pose.translate(y=-0.001)
            elif key == ARROW_DOWN_KEY:
                pose.translate(y=0.001)
            elif key == ARROW_LEFT_KEY:
                pose.rotate(y=math.radians(-1))
            elif key == ARROW_RIGHT_KEY:
                pose.rotate(y=math.radians(1))
    if not show:
        np.save(os.path.join(SEQUENCE_PATH, "poses.npy"), np.array(fixed_poses))


