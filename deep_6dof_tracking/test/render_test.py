import os

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
    preload_count = 0
    # Iterate over all models from config files
    geometry_path = os.path.join(MODEL["path"], "geometry.ply")
    vpRender = ModelRenderer(geometry_path, SHADER_PATH, camera, window, window_size)
    OBJECT_WIDTH = int(MODEL["width"])

    posttransforms = Compose([ GaussianNoise(2, 20, proba=1),
                               #GaussianBlur(3, min_kernel_size=3, proba=1)
                               ])

    z_distance = 1.2

    for z_distance in [1.2, 0.9, 0.7, 0.4]:
        object_pose = Transform.from_parameters(0, 0, -z_distance, 0, 0, 0)

        value_index = [3]
        #values = np.linspace(-0.025, 0.025, STEPS)
        values = np.linspace(math.radians(-50), math.radians(50), STEPS)
        errors_no_augment = []
        pixels_diff = []
        images = []
        labels = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
        label = " ".join([x for i, x in enumerate(labels) if i in value_index])
        for v in tqdm(values):
            v_array = np.zeros(6)
            v_array[value_index] = v
            pair_transform = Transform.from_parameters(*v_array)
            pair = combine_view_transform(object_pose, pair_transform)

            rgbA, depthA = vpRender.render_image(object_pose)
            rgbB, depthB = vpRender.render_image(pair, sphere_sampler.random_direction())
            bb = compute_2Dboundingbox(object_pose, camera, OBJECT_WIDTH, scale=(1000, -1000, -1000))
            rgbA, depthA = normalize_scale(rgbA, depthA, bb, IMAGE_SIZE)
            rgbB, depthB = normalize_scale(rgbB, depthB, bb, IMAGE_SIZE)

            maskA = depthA != 0
            maskB = depthB != 0
            diff_mask = np.logical_xor(maskA, maskB)
            non_diff_mask = np.logical_not(diff_mask)
            pixels_diff.append(np.sum(diff_mask))

            depth_diff = depthA - depthB
            error = np.abs(depth_diff)
            error *= non_diff_mask
            non_zero_error = error[error != 0]
            mean_error = np.mean(non_zero_error)
            errors_no_augment.append(mean_error)

            # mask depth
            #depth_diff *= non_diff_mask
            images.append(depth_diff)

            #augment
            """
            mask = (depthB != 0).astype(np.int)
            data = (rgbA, depthA, rgbB, depthB, object_pose.to_parameters())
            data = posttransforms(data)
            rgbA, depthA, rgbB, depthB, _ = data
            depthB *= mask
    
            error = np.mean(np.abs(depthA - depthB))
            errors_augment.append(error)
            """
            if DEBUG:
                print("Delta T :\n{}".format(pair_transform))
                #fig, axis = plt.subplots(2, 2)
                #ax1, ax2 = axis[0, :]
                #ax3, ax4 = axis[1, :]
                #ax1.imshow(diff_mask.astype(np.uint8)*255)
                show_frames(rgbA, depthA, rgbB, depthB)
            cv2.imshow("test", np.concatenate((rgbA[:, :, ::-1], rgbB[:, :, ::-1]), axis=1))
            k = cv2.waitKey(1)
            if k == ESCAPE_KEY:
                break

        mosaic = np.concatenate(images, axis=1)
        #plt.imshow(mosaic, vmin=-20, vmax=20)
        plt.imshow(mosaic)
        width = IMAGE_SIZE[0]
        plt.xticks(np.arange(width/2, width*len(values), width), ["{:10.3f}".format(x) for x in values])
        plt.xlabel("{} Delta".format(label))
        plt.title("depth difference w.r.t transform. (object distance : {}m)".format(z_distance))
        plt.show()

        pixels_diff = np.array(pixels_diff)
        plt.subplot("122")
        plt.plot(values, pixels_diff)

        errors_no_augment = np.array(errors_no_augment)
        plt.subplot("121")
        plt.plot(values, errors_no_augment)
        #plt.legend(["No augmentation", "augmentation"])
        plt.xlabel("Z Delta")
        plt.ylabel("Mean pixel error")
        plt.title("Z distance = {}".format(z_distance))

        plt.show()
