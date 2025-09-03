from deep_6dof_tracking.utils.transform import Transform
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
from deep_6dof_tracking.data.model_rend_test import ModelRenderer3
from tqdm import tqdm
import argparse
import shutil
import os
import math
import numpy as np
import cv2
ESCAPE_KEY = 1048603

import matplotlib.pyplot as plt
import trimesh
from trimesh import Geometry

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('-c', '--camera', help="camera json path", action="store",
                        default="../data/sensors/camera_parameter_files/synthetic.json")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="models folder path", action="store", default="./model_config.yaml")

    parser.add_argument('-y', '--saveformat', help="save format", action="store", default="numpy")

    parser.add_argument('-s', '--samples', help="quantity of samples", action="store", default=200000, type=int)
    parser.add_argument('-t', '--translation', help="max translation in meter", action="store", default=0.02, type=float)
    parser.add_argument('-r', '--rotation', help="max rotation in degree", action="store", default=20, type=float)
    parser.add_argument('-e', '--resolution', help="image pixel size", action="store", default=150, type=int)
    parser.add_argument('--boundingbox', help="Crop bounding box width in %% of the object max width",
                        action="store", default=10, type=int)

    parser.add_argument('-a', '--maxradius', help="max distance", action="store", default=1.4, type=float)
    parser.add_argument('-i', '--minradius', help="min distance", action="store", default=0.8, type=float)

    parser.add_argument('-p', '--preload', help="Load any data saved in output directory", action="store_true")
    parser.add_argument('-v', '--show', help="show image while generating", action="store_true")
    parser.add_argument('-d', '--debug', help="show debug screen while generating", action="store_true")
    parser.add_argument('--random_sampling', help="sample displacement vector uniformly", action="store_true")
    parser.add_argument('--specular', help="Will add random specularity to the training set", action="store_true")
    parser.add_argument('--depthonly', help="Only generate depth data", action="store_true")
    parser.add_argument('--no_norm', help='dont do size normalization on objects', action='store_true')


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
    SPECULAR = arguments.specular

    SHADER_PATH = arguments.shader
    OUTPUT_PATH = arguments.output
    CAMERA_PATH = arguments.camera
    BOUNDING_BOX = arguments.boundingbox
    DEPTHONLY = arguments.depthonly
    NO_NORM = arguments.no_norm

    #MODELS = yaml_load(arguments.model)["models"]
    models_path = arguments.model
    models = os.listdir(arguments.model)


    # if SHOW:
    #     import cv2
    # import cv2
    # if not os.path.exists(OUTPUT_PATH):
    #     os.mkdir(OUTPUT_PATH)
    # shutil.copy(arguments.model, os.path.join(OUTPUT_PATH, "models.yml"))

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
    metadata["bounding_box"] = float(BOUNDING_BOX)/100
    for m in models:
        metadata[m] = {}
        metadata[m]["translation_range"] = str(TRANSLATION_RANGE)
        metadata[m]["rotation_range"] = str(ROTATION_RANGE)
        metadata[m]["image_size"] = str(IMAGE_SIZE[0])
        metadata[m]["save_type"] = SAVE_TYPE
        metadata[m]["object_width"] = {}
        metadata[m]["min_radius"] = str(SPHERE_MIN_RADIUS)
        metadata[m]["max_radius"] = str(SPHERE_MAX_RADIUS)
        TRANSLATION_VARIANCE = 0.001
        ROTATION_VARIANCE = 0.05
        metadata[m]["translation_variance"] = TRANSLATION_VARIANCE
        metadata[m]["rotation_variance"] = ROTATION_VARIANCE

    camera = Camera.load_from_json(CAMERA_PATH)
    dataset = DeepTrackLoaderBase(OUTPUT_PATH, read_data=PRELOAD)
    dataset.set_save_type(SAVE_TYPE)
    dataset.camera = camera
    window_size = (camera.width, camera.height)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    preload_count = 0
    print("Compute Mean bounding box")
    widths = {}
    true_widths = {}
    print(models)
    for model in tqdm(models):
        current_model = os.path.join(models_path, model)
        files = os.listdir(current_model)
        print(current_model)
        print(current_model)
        print(files)
        for file in files:
            if file.endswith(".obj"):
                print(file)
                print(file)
                geometry_path = os.path.join(current_model, file)
                mesh : Geometry = trimesh.load(geometry_path)
                model_3d = mesh.vertices
                object_max_width = maximum_width(model_3d) * 1000
                metadata[model]["object_width_individual"] = str(object_max_width)
                true_widths[model] = object_max_width
                with_add = BOUNDING_BOX / 100 * object_max_width
                widths[model] = object_max_width + with_add
    
    true_sorted = list(true_widths.values())
    true_sorted.sort()
    print(true_sorted)
    median_width = true_sorted[int(len(true_sorted)/2)]
    print(median_width)

    #widths.sort()
    sorted = list(widths.values())
    sorted.sort()
    print(sorted)
    OBJECT_WIDTH = sorted[int(len(sorted)/2)]
    metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    if not NO_NORM:
        metadata["median_width"] = str(median_width)
    for model in models:
        metadata[model]["bounding_box_width"] = str(OBJECT_WIDTH)
    #OBJECT_WIDTH = 220
    print(models)
    
    # metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    print("Mean object width : {}".format(OBJECT_WIDTH))
    # Iterate over all models from config files
    for model in models:
        print(model)
        current_model = os.path.join(models_path, model)
        files = os.listdir(current_model)
        for file in files:
            if file.endswith(".obj"):
                geometry_path = os.path.join(current_model, file)
            elif file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                texture_path = os.path.join(current_model, file)

        print(geometry_path)
        # ao_path = os.path.join(model["path"], "ao.ply")
        # WARNING : OBJECT_WIDTH is not the max width with multiple objects
        # vpRender = ModelRenderer2(geometry_path, SHADER_PATH, dataset.camera, [window_size, IMAGE_SIZE], object_max_width=OBJECT_WIDTH)
        # if os.path.exists(ao_path):
        #     vpRender.load_ambiant_occlusion_map(ao_path)
        try:
            ratio1 = median_width / float(metadata[model]["object_width_individual"])
        except:
            ratio1 = None
        print(median_width)
        print(float(metadata[model]["object_width_individual"]))
        print(f'ratio1: {ratio1}')

        median_width = None if NO_NORM else median_width

        vpRender = ModelRenderer3(geometry_path,
                                  SHADER_PATH,
                                  texture_path,
                                  dataset.camera,
                                  [window_size, IMAGE_SIZE],
                                  original_width=float(metadata[model]["object_width_individual"]),
                                  new_width=median_width)
        for i in tqdm(range(SAMPLE_QUANTITY - preload_count)):
            random_pose = sphere_sampler.get_random()

            if RANDOM_SAMPLING:
                # Sampling from uniform distribution in the ranges
                random_transform = Transform.random((-TRANSLATION_RANGE, TRANSLATION_RANGE),
                                                    (-ROTATION_RANGE, ROTATION_RANGE))
            else:
                # Sampling from gaussian ditribution in the magnitudes
                random_transform = sphere_sampler.random_normal_magnitude(TRANSLATION_RANGE, ROTATION_RANGE)

            #random_transform = Transform.from_parameters(0, 0, 0, 0, 0, 0)
            pair = combine_view_transform(random_pose, random_transform)
            bb = compute_2Dboundingbox(random_pose, dataset.camera, OBJECT_WIDTH, scale=(1000, 1000, -1000))
            left = np.min(bb[:, 1])
            right = np.max(bb[:, 1])
            top = np.min(bb[:, 0])
            bottom = np.max(bb[:, 0])
            vpRender.setup_camera(camera, left, right, bottom, top)
            rgbA, depthA = vpRender.render_image(random_pose, fbo_index=1)

            light_intensity = np.zeros(3)
            light_intensity.fill(np.random.uniform(0.1, 1.3))
            light_intensity += np.random.uniform(-0.1, 0.1, 3)
            ambiant_light = np.zeros(3)
            ambiant_light.fill(np.random.uniform(0.5, 0.75))
            shininess = 0
            if np.random.randint(0, 2) and SPECULAR:
                shininess = np.random.uniform(3, 30)
            vpRender.setup_camera(camera, 0, camera.width, camera.height, 0)
            rgbB, depthB = vpRender.render_image(pair,
                                                 fbo_index=0,
                                                 light_direction=sphere_sampler.random_direction(),
                                                 light_diffuse=light_intensity,
                                                 ambiant_light=ambiant_light,
                                                 shininess=shininess)
            rgbB, depthB = normalize_scale(rgbB, depthB, bb, IMAGE_SIZE)

            index = dataset.add_pose(rgbA, depthA, random_pose, depthonly=DEPTHONLY)
            dataset.add_pair(rgbB, depthB, random_transform, index, depthonly=DEPTHONLY)

            if i % 500 == 0:
                dataset.dump_images_on_disk()
            if i % 5000 == 0:
                dataset.save_json_files(metadata)

            if DEBUG:
                show_frames(rgbA, depthA, rgbB, depthB)
            if SHOW:
                cv2.imshow("test", np.concatenate((rgbA[:, :, ::-1], rgbB[:, :, ::-1]), axis=1))
                k = cv2.waitKey(1)
                if k == ESCAPE_KEY:
                    break

    dataset.dump_images_on_disk()
    dataset.save_json_files(metadata)
