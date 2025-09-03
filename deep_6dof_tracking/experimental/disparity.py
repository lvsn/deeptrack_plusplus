from pytorch_toolbox.io import yaml_load
from pytorch_toolbox.transformations.compose import Compose

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

ESCAPE_KEY = 1048603


def compute_disparity(points, camera, disparity):
    points[:, 0] = points[:, 0].clip(0, camera.height - 1)
    points[:, 1] = points[:, 1].clip(0, camera.width - 1)
    disparity_map = np.zeros((camera.height, camera.width, 2))
    disparity_map[points[:, 0], points[:, 1], 0] = disparity[:, 0]
    disparity_map[points[:, 0], points[:, 1], 1] = disparity[:, 1]
    return disparity_map

if __name__ == '__main__':

    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "disparity_config.yml"
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
    preload_count = 0
    # Iterate over all models from config files
    vpRender = ModelRenderer(MODEL["geometry_path"], SHADER_PATH, camera, window, window_size)
    vpRender.load_ambiant_occlusion_map(MODEL["ao_path"])
    OBJECT_WIDTH = int(MODEL["width"])

    posttransforms = Compose([ GaussianNoise(2, 20, proba=1),
                               #GaussianBlur(3, min_kernel_size=3, proba=1)
                               ])

    z_distance = 0.5
    object_pose = Transform.from_parameters(0, 0, -z_distance, 0, 0, 0)

    value_index = [5]
    #values = np.linspace(-0.03, 0.03, STEPS)
    values = np.linspace(math.radians(10), math.radians(-10), STEPS)
    errors_no_augment = []
    pixels_diff = []
    images = []
    labels = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
    label = " ".join([x for i, x in enumerate(labels) if i in value_index])
    for v in tqdm(values):
        v_array = np.zeros(6)
        v_array[value_index] = v
        #pair_transform = Transform.random((-0.02, 0.02), (0, 0))
        #pair_transform = Transform.random((0, 0), (math.radians(-20), math.radians(20)))
        #pair_transform = Transform.random((-0.02, 0.02), (math.radians(-10), math.radians(10)))
        pair_transform = Transform.from_parameters(*v_array)
        print("---")
        print(pair_transform)
        pair = combine_view_transform(object_pose, pair_transform)

        rgbA_orig, depthA_orig = vpRender.render_image(object_pose)
        rgbB_orig, depthB_orig = vpRender.render_image(pair, sphere_sampler.random_direction())
        bb = compute_2Dboundingbox(object_pose, camera, OBJECT_WIDTH, scale=(1000, -1000, -1000))
        rgbA, depthA = normalize_scale(rgbA_orig, depthA_orig, bb, IMAGE_SIZE)
        rgbB, depthB = normalize_scale(rgbB_orig, depthB_orig, bb, IMAGE_SIZE)

        # backproject depth A
        # transform points
        # reproject

        pointsA = camera.backproject_depth(depthA_orig)/1000
        pointsA = pointsA[pointsA.all(axis=1)]

        # backproject and transform points
        pointsB = object_pose.dot(pointsA)
        pointsB = pair_transform.dot(pointsB)
        pointsB = object_pose.inverse().dot(pointsB)

        # project points
        pointsB_2D = camera.project_points(pointsB)
        pointsA_2D = camera.project_points(pointsA)
        disparityA = pointsA_2D - pointsB_2D
        disparityB = pointsB_2D - pointsA_2D
        #print(disparity.min(axis=0), disparity.max(axis=0))
        depthB_rec = compute_disparity(pointsB_2D, camera, disparityB)
        depthA_rec = compute_disparity(pointsA_2D, camera, disparityA)

        rgbB, disparityB_y = normalize_scale(rgbB_orig, depthB_rec[:, :, 0], bb, IMAGE_SIZE)
        rgbA, disparityA_y = normalize_scale(rgbA_orig, depthA_rec[:, :, 0], bb, IMAGE_SIZE)
        rgbB, disparityB_x = normalize_scale(rgbB_orig, depthB_rec[:, :, 1], bb, IMAGE_SIZE)
        rgbA, disparityA_x = normalize_scale(rgbA_orig, depthA_rec[:, :, 1], bb, IMAGE_SIZE)

        fig, axes = plt.subplots(3, 2)
        axes[0, 0].imshow(depthA)
        axes[0, 1].imshow(depthB)
        cmap_min_thresh = None
        cmap_max_thresh = None
        axes[1, 0].imshow(disparityA_y[:, :], vmin=cmap_min_thresh, vmax=cmap_max_thresh)
        axes[1, 1].imshow(disparityB_y[:, :], vmin=cmap_min_thresh, vmax=cmap_max_thresh)
        axes[2, 0].imshow(disparityA_x[:, :], vmin=cmap_min_thresh, vmax=cmap_max_thresh)
        axes[2, 1].imshow(disparityB_x[:, :], vmin=cmap_min_thresh, vmax=cmap_max_thresh)
        plt.show()