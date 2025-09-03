from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.data.data_augmentation import HSVNoise, Background, GaussianNoise, GaussianBlur, OffsetDepth
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    camera = Camera.load_from_json(CAMERA_PATH)
    window_size = (camera.width, camera.height)
    window = InitOpenGL(*window_size)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    vpRender = ModelRenderer(MODEL["geometry_path"], SHADER_PATH, camera, window, window_size)
    vpRender.load_ambiant_occlusion_map(MODEL["ao_path"])
    OBJECT_WIDTH = int(MODEL["width"])

    object_pose = sphere_sampler.get_random()
    light_direction = sphere_sampler.random_direction()
    light_intensity = np.zeros(3)
    light_intensity.fill(random.uniform(0.4, 4))
    light_intensity += np.random.uniform(-0.5, 0.5, 3)

    render, depth = vpRender.render_image(object_pose, light_direction, light_intensity)

    cv2.imshow("render", render[:, :, ::-1])
    cv2.waitKey()
