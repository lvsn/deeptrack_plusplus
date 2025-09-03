from pytorch_toolbox.io import yaml_load
from pytorch_toolbox.transformations.compose import Compose

from deep_6dof_tracking.data.data_augmentation import HSVNoise, Background, GaussianNoise, GaussianBlur, OffsetDepth
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ESCAPE_KEY = 1048603

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
    CAMERA_PATH2 = configs["camera_path_2"]
    DEBUG = configs["debug"]

    camera = Camera.load_from_json(CAMERA_PATH)
    camera2 = Camera.load_from_json(CAMERA_PATH2)
    window_size = (camera.width, camera.height)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    preload_count = 0
    # Iterate over all models from config files
    vpRender = ModelRenderer2(MODEL["geometry_path"], SHADER_PATH, camera, [window_size, (150, 150)])
    vpRender.load_ambiant_occlusion_map(MODEL["ao_path"])
    OBJECT_WIDTH = int(MODEL["width"])

    posttransforms = Compose([GaussianNoise(2, 20, proba=1),
                              # GaussianBlur(3, min_kernel_size=3, proba=1)
                              ])

    z_distance = 1.2
    object_pose = Transform.from_parameters(0, 0, -z_distance, 0, 0, 0)
    bb = compute_2Dboundingbox(object_pose, camera, OBJECT_WIDTH, scale=(1000, -1000, -1000))
    left = np.min(bb[:, 1])
    right = np.max(bb[:, 1])
    top = np.min(bb[:, 0])
    bottom = np.max(bb[:, 0])

    vpRender.setup_camera(camera, left, right, bottom, top)
    rgb, depth = vpRender.render_image(object_pose, fbo_index=1)
    #rgb, depth = normalize_scale(rgb, depth, bb, IMAGE_SIZE)

    cv2.imshow("test_crop", rgb[:, :, ::-1])

    bb = compute_2Dboundingbox(object_pose, camera2, OBJECT_WIDTH, scale=(1000, -1000, -1000))
    left = np.min(bb[:, 1])
    right = np.max(bb[:, 1])
    top = np.min(bb[:, 0])
    bottom = np.max(bb[:, 0])
    vpRender.setup_camera(camera2, left, right, bottom, top)
    rgb_gpu, depth_gpu = vpRender.render_image(object_pose, fbo_index=1)

    plt.subplot("121")
    plt.imshow(np.concatenate((depth, depth_gpu), axis=1))
    plt.subplot("122")
    plt.imshow(depth.astype(np.float32) - depth_gpu.astype(np.float32))
    plt.show()
