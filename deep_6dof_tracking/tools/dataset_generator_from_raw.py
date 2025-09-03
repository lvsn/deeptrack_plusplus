import argparse

from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.data.sequence_loader import SequenceLoader
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import combine_view_transform, center_pixel, show_frames, compute_2Dboundingbox, \
    normalize_scale, project_center
from deep_6dof_tracking.utils.camera import Camera
from pytorch_toolbox.io import yaml_load
from scipy import ndimage
import sys
import os
import math
import cv2
import numpy as np
import random

from tqdm import tqdm

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler

ESCAPE_KEY = 1048603


def mask_real_image(color, depth, depth_render):
    mask = (depth_render != 0).astype(np.uint8)[:, :, np.newaxis]
    masked_rgb = color * mask

    masked_hsv = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2HSV)
    saturation_mask = (masked_hsv[:, :, 2] <= SATURATION_THRESHOLD)[:, :, np.newaxis].astype(np.uint8)
    total_mask = np.bitwise_and(mask, saturation_mask)

    masked_color = color * total_mask
    masked_depth = depth[:total_mask.shape[0], :total_mask.shape[1]] * total_mask[:, :, 0]
    return masked_color, masked_depth


def random_z_rotation(rgb, depth, pose, camera):
    rotation = random.uniform(-180, 180)
    rotation_matrix = Transform()
    rotation_matrix.set_rotation(0, 0, math.radians(rotation))

    pixel = center_pixel(pose, camera)
    new_rgb = rotate_image(rgb, rotation, pixel[0])
    new_depth = rotate_image(depth, rotation, pixel[0])
    # treshold below 50 means we remove some interpolation noise, which cover small holes
    #mask = (new_depth >= 50).astype(np.uint8)[:, :, np.newaxis]
    #rgb_mask = np.all(new_rgb != 0, axis=2).astype(np.uint8)
    #kernel = np.array([[0, 1, 0],
    #                   [1, 1, 1],
    #                   [0, 1, 0]], np.uint8)
    # erode rest of interpolation noise which will affect negatively future blendings
    #eroded_mask = cv2.erode(mask, kernel, iterations=2)
    #eroded_rgb_mask = cv2.erode(rgb_mask, kernel, iterations=2)
    #new_depth = new_depth * eroded_mask
    #new_rgb = new_rgb * eroded_rgb_mask[:, :, np.newaxis]
    new_pose = combine_view_transform(pose, rotation_matrix)
    return new_rgb, new_depth, new_pose


def rotate_image(img, angle, pivot):
    pivot = pivot.astype(np.int32)
    # double size of image while centering object
    pads = [[img.shape[0] - pivot[0], pivot[0]], [img.shape[1] - pivot[1], pivot[1]]]
    if len(img.shape) > 2:
        pads.append([0, 0])
    imgP = np.pad(img, pads, 'constant')
    # reduce size of matrix to rotate around the object
    if len(img.shape) > 2:
        total_y = np.sum(imgP.any(axis=(0, 2))) * 2.4
        total_x = np.sum(imgP.any(axis=(1, 2))) * 2.4
    else:
        total_y = np.sum(imgP.any(axis=0)) * 2.4
        total_x = np.sum(imgP.any(axis=1)) * 2.4
    cropy = int((imgP.shape[0] - total_y)/2)
    cropx = int((imgP.shape[1] - total_x)/2)
    imgP[cropy:-cropy, cropx:-cropx] = ndimage.rotate(imgP[cropy:-cropy, cropx:-cropx], angle,
                                                      reshape=False, prefilter=False)

    return imgP[pads[0][0]: -pads[0][1], pads[1][0]: -pads[1][1]]

if __name__ == '__main__':

    #
    #   Load configurations
    #
    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="model file path", action="store", default="./model_config.yaml")

    parser.add_argument('-y', '--saveformat', help="save format", action="store", default="numpy")

    parser.add_argument('-s', '--samples', help="quantity of samples", action="store", default=200000, type=int)
    parser.add_argument('-t', '--translation', help="max translation in meter", action="store", default=0.02, type=float)
    parser.add_argument('-r', '--rotation', help="max rotation in degree", action="store", default=20, type=float)
    parser.add_argument('-e', '--resolution', help="image pixel size", action="store", default=150, type=int)
    parser.add_argument('--boundingbox', help="Crop bounding box width in % of the object max width",
                        action="store", default=10, type=int)

    parser.add_argument('-a', '--maxradius', help="max distance", action="store", default=1.4, type=float)
    parser.add_argument('-i', '--minradius', help="min distance", action="store", default=0.8, type=float)

    parser.add_argument('-p', '--preload', help="Load any data saved in output directory", action="store_true")
    parser.add_argument('-v', '--show', help="show image while generating", action="store_true")
    parser.add_argument('-d', '--debug', help="show debug screen while generating", action="store_true")
    parser.add_argument('--random_sampling', help="show debug screen while generating", action="store_true")

    parser.add_argument('--real_path', help="Path to real raw data", action="store", default="")
    parser.add_argument('--saturation_threshold', help="Will remove pixels over this saturation value", action="store", default=240, type=int)


    arguments = parser.parse_args()

    TRANSLATION_RANGE = arguments.translation
    ROTATION_RANGE = math.radians(arguments.rotation)
    SAMPLE_QUANTITY = arguments.samples
    SPHERE_MIN_RADIUS = arguments.minradius
    SPHERE_MAX_RADIUS = arguments.maxradius
    IMAGE_SIZE = (arguments.resolution, arguments.resolution)
    PRELOAD = arguments.preload
    SAVE_TYPE = arguments.saveformat
    SHOW = arguments.show
    DEBUG = arguments.debug
    RANDOM_SAMPLING = arguments.random_sampling

    SHADER_PATH = arguments.shader
    OUTPUT_PATH = arguments.output
    BOUNDING_BOX = arguments.boundingbox
    REAL_PATH = arguments.real_path
    SATURATION_THRESHOLD = arguments.saturation_threshold
    MODELS = yaml_load(arguments.model)["models"]

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Write important misc data to file
    metadata = {}
    metadata["translation_range"] = str(TRANSLATION_RANGE)
    metadata["rotation_range"] = str(ROTATION_RANGE)
    metadata["image_size"] = str(IMAGE_SIZE[0])
    metadata["save_type"] = SAVE_TYPE
    metadata["object_width"] = {}
    metadata["min_radius"] = str(SPHERE_MIN_RADIUS)
    metadata["max_radius"] = str(SPHERE_MAX_RADIUS)
    TRANSLATION_VARIANCE = 0.001
    ROTATION_VARIANCE = 0.05
    metadata["translation_variance"] = TRANSLATION_VARIANCE
    metadata["rotation_variance"] = ROTATION_VARIANCE

    real_dataset = SequenceLoader(REAL_PATH)
    camera = Camera.load_from_json(real_dataset.root)
    output_dataset = DeepTrackLoaderBase(OUTPUT_PATH, read_data=True)
    output_dataset.set_save_type(SAVE_TYPE)
    output_dataset.camera = camera
    window_size = (real_dataset.camera.width, real_dataset.camera.height)
    print(window_size)

    geometry_path = os.path.join(MODELS[0]["path"], "geometry.ply")
    ao_path = os.path.join(MODELS[0]["path"], "ao.ply")

    model_3d = PlyParser(geometry_path).get_vertex()
    object_max_width = maximum_width(model_3d) * 1000
    OBJECT_WIDTH = (BOUNDING_BOX / 100 * object_max_width) + object_max_width

    metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    print("Object width : {}".format(OBJECT_WIDTH))

    vpRender = ModelRenderer2(geometry_path, SHADER_PATH, real_dataset.camera, [window_size, IMAGE_SIZE])
    vpRender.load_ambiant_occlusion_map(ao_path)
    per_object_samples = max(1, int(SAMPLE_QUANTITY / real_dataset.size()))
    for i in range(real_dataset.size()):
        frame, pose = real_dataset.data_pose[i]

        vpRender.setup_camera(real_dataset.camera, 0, real_dataset.camera.width, real_dataset.camera.height, 0)
        rgb_render, depth_render = vpRender.render_image(pose)
        rgb, depth = frame.get_rgb_depth(real_dataset.root)
        masked_rgb, masked_depth = mask_real_image(rgb, depth, depth_render)

        print("image {} of {}".format(i, real_dataset.size()))
        for j in tqdm(range(per_object_samples)):
            rotated_rgb, rotated_depth, rotated_pose = random_z_rotation(masked_rgb, masked_depth, pose, real_dataset.camera)
            if RANDOM_SAMPLING:
                # Sampling from uniform distribution in the ranges
                random_transform = Transform.random((-TRANSLATION_RANGE, TRANSLATION_RANGE),
                                                    (-ROTATION_RANGE, ROTATION_RANGE))
            else:
                # Sampling from gaussian ditribution in the magnitudes
                random_transform = UniformSphereSampler.random_normal_magnitude(TRANSLATION_RANGE, ROTATION_RANGE)

            inverted_random_transform = Transform.from_parameters(*(-random_transform.to_parameters()))
            object_B_pose = rotated_pose.copy()
            previous_pose = combine_view_transform(object_B_pose, inverted_random_transform)
            bb = compute_2Dboundingbox(previous_pose, real_dataset.camera, OBJECT_WIDTH, scale=(1000, 1000, -1000))
            left = np.min(bb[:, 1])
            right = np.max(bb[:, 1])
            top = np.min(bb[:, 0])
            bottom = np.max(bb[:, 0])
            vpRender.setup_camera(real_dataset.camera, left, right, bottom, top)
            rgbA, depthA = vpRender.render_image(previous_pose, fbo_index=1)

            bb = compute_2Dboundingbox(previous_pose, real_dataset.camera, OBJECT_WIDTH, scale=(1000, -1000, -1000))
            rgbB, depthB = normalize_scale(rotated_rgb, rotated_depth, bb, IMAGE_SIZE)

            index = output_dataset.add_pose(rgbA, depthA, previous_pose)
            output_dataset.add_pair(rgbB, depthB, random_transform, index)
            iteration = i * per_object_samples + j

            if iteration % 500 == 0:
                output_dataset.dump_images_on_disk()
            if iteration % 5000 == 0:
                output_dataset.save_json_files(metadata)

            if DEBUG:
                show_frames(rgbA, depthA, rgbB, depthB)
            cv2.imshow("testB", np.concatenate((rgbA[:, :, ::-1], rgbB[:, :, ::-1]), axis=1))
            k = cv2.waitKey(1)
            if k == ESCAPE_KEY:
                break

    output_dataset.dump_images_on_disk()
    output_dataset.save_json_files(metadata)
