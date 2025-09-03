from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
import os
import sys
import math
import cv2
import numpy as np
import random
ESCAPE_KEY = 1048603
ARROW_UP = 1113938
ARROW_DOWN = 1113940
ARROW_LEFT = 1113937
ARROW_RIGHT = 1113939


if __name__ == '__main__':

    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "dataset_test_config.yml"
    configs = yaml_load(config_path)

    # Populate important data from config file
    MODEL = configs["model"]
    SHADER_PATH = configs["shader_path"]
    STEPS = configs["steps"]
    TRANSLATION_RANGE = configs["translation_range"]
    ROTATION_RANGE = math.radians(configs["rotation_range"])
    SPHERE_MIN_RADIUS = configs["sphere_min_radius"]
    SPHERE_MAX_RADIUS = configs["sphere_max_radius"]
    IMAGE_SIZE = (configs["image_size"], configs["image_size"])
    CAMERA_PATH = configs["camera_path"]
    DEBUG = configs["debug"]
    shininess = 0

    camera = Camera.load_from_json(CAMERA_PATH)
    window_size = (camera.width, camera.height)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    preload_count = 0
    # Iterate over all models from config files
    vpRender = ModelRenderer2(os.path.join(MODEL["path"], "geometry.ply"), SHADER_PATH, camera, [window_size])
    if os.path.exists(os.path.join(MODEL["path"], "ao.ply")):
        vpRender.load_ambiant_occlusion_map(os.path.join(MODEL["path"], "ao.ply"))
    OBJECT_WIDTH = int(MODEL["width"])

    object_pose = sphere_sampler.get_random()
    light_direction = sphere_sampler.random_direction()
    light_intensity = np.zeros(3)
    light_intensity.fill(random.uniform(0.4, 4))
    light_intensity += np.random.uniform(-0.5, 0.5, 3)

    while True:

        render, depth = vpRender.render_image(object_pose, 0, light_direction, light_intensity, shininess=shininess)
        bb = compute_2Dboundingbox(object_pose, camera, OBJECT_WIDTH, scale=(1000, -1000, -1000))
        network_rgb, network_depth = normalize_scale(render, depth, bb, IMAGE_SIZE)

        #plt.subplot("111")
        #plt.imshow(render)
        #plt.show()
        cv2.imshow("render", render[:, :, ::-1])

        key = cv2.waitKey(30)
        key_chr = chr(key & 255)
        if key == ESCAPE_KEY:
            break
        if key_chr == 'p':
            object_pose = sphere_sampler.get_random()
        if key_chr == 'l':
            light_direction = sphere_sampler.random_direction()
            light_intensity = np.zeros(3)
            light_intensity.fill(random.uniform(0.4, 4))
            light_intensity += np.random.uniform(-0.5, 0.5, 3)
        if key_chr == 'q':
            light_intensity -= 0.25
            light_intensity = light_intensity.clip(0, 100)
        if key_chr == 'w':
            light_intensity += 0.25
            light_intensity = light_intensity.clip(0, 100)
        if key_chr == 'a':
            shininess = max(shininess - 2, 0)
        if key_chr == 's':
            shininess += 2
        if key == ARROW_LEFT:
            delta = Transform.from_parameters(0, 0, 0, 0, -10, 0, is_degree=True)
            object_pose = combine_view_transform(object_pose, delta)
        if key == ARROW_RIGHT:
            delta = Transform.from_parameters(0, 0, 0, 0, 10, 0, is_degree=True)
            object_pose = combine_view_transform(object_pose, delta)
        if key == ARROW_DOWN:
            delta = Transform.from_parameters(0, 0, 0, -10, 0, 0, is_degree=True)
            object_pose = combine_view_transform(object_pose, delta)
        if key == ARROW_UP:
            delta = Transform.from_parameters(0, 0, 0, 10, 0, 0, is_degree=True)
            object_pose = combine_view_transform(object_pose, delta)
        if key_chr == " ":
            print(object_pose)


        if key != -1:
            print("Key pressed : {} id: {}".format(key_chr, key))
