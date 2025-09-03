"""
    Example script to load a sequence and a 3D model
"""
import json

import cv2
import numpy as np
import os

from PIL import Image
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform

GREEN_HUE = 120 / 2
RED_HUE = 230 / 2
BLUE_HUE = 0
LIGH_BLUE_HUE = 20
PURPLE_HUE = 300 / 2

BRIGHTNESS = 100

class SequenceLoader:
    def __init__(self, root):
        self.root = root
        self.data_pose = None
        self.current = 0

        with open(os.path.join(self.root, "meta_data.json")) as data_file:
            data = json.load(data_file)
        self.camera = Camera.load_from_json(self.root)
        self.metadata = data["metaData"]
        self.poses = np.load(os.path.join(self.root, "poses.npy"))

    def get_frame(self, i):
        pose = Transform.from_matrix(self.poses[i].reshape(4, 4))
        rgb = np.array(Image.open(os.path.join(self.root, "{}.png".format(i))))
        depth = np.array(Image.open(os.path.join(self.root, "{}d.png".format(i)))).astype(np.uint16)
        return pose, rgb, depth

    def size(self):
        return len(self.poses)

    def __iter__(self):
        self.current = 0
        return self

    def __len__(self):
        return self.size()

    def __getitem__(self, i):
        return self.get_frame(i)

    def __next__(self):
        if self.current > self.size():
            raise StopIteration
        else:
            self.current += 1
            return self.get_frame(self.current - 1)

def set_hue(rgb_input, hue_value):
    hsv = cv2.cvtColor(rgb_input, cv2.COLOR_RGB2HSV).astype(int)
    hsv[:, :, 0] = hue_value
    hsv[:, :, 1] += 150
    hsv[:, :, 2] += 100
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    rgb_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb_output[rgb_input == 0] = 0
    return rgb_output


def image_blend_gray(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    will set the background gray
    :param foreground:
    :param background:
    :return:
    """
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    background_gray[:, :] = (np.clip(background_gray.astype(int) - BRIGHTNESS, 0, 255)).astype(np.uint8)
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, :] == 0
        mask = np.all(mask, axis=2)[:, :, np.newaxis]
    return background_gray[:, :, np.newaxis] * mask + foreground


if __name__ == '__main__':

    model_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/tide/geometry.ply"
    sequence_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/choi_christensen/tide"
    shader_path = "../data/shaders"

    sequence = SequenceLoader(sequence_path)
    renderer = ModelRenderer2(model_path, shader_path, sequence.camera,
                             [(sequence.camera.width, sequence.camera.height)])

    for pose, rgb, depth in sequence:
        rgb_render, depth_render = renderer.render_image(pose)
        rgb_render = set_hue(rgb_render, PURPLE_HUE)

        rgb = cv2.pyrDown(rgb)
        rgb_render = cv2.pyrDown(rgb_render)

        blend = image_blend_gray(rgb_render[:, :, ::-1], rgb)

        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("blending", blend)
        cv2.waitKey(10)
