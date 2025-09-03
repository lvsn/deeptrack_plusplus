import configparser
from deep_6dof_tracking.utils.transform import Transform
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
from tqdm import tqdm
import argparse
import shutil
import os
import math
import numpy as np
ESCAPE_KEY = 1048603

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('-c', '--camera', help="camera json path", action="store",
                        default="../data/sensors/camera_parameter_files/synthetic.json")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="model file path", action="store", default="./model_config.yaml")

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

    parser.add_argument('--config', help="config file path", action="store", default=None)

    arguments = parser.parse_args()

    if arguments.config is not None:
        config = configparser.ConfigParser()
        config.read(arguments.config)
        TRANSLATION_RANGE = float(config["DEFAULT"]["translation"])
        ROTATION_RANGE = math.radians(float(config["DEFAULT"]["rotation"]))
        SAMPLE_QUANTITY = int(config["DEFAULT"]["samples"])
        IMAGE_SIZE = (int(config["DEFAULT"]["resolution"]), int(config["DEFAULT"]["resolution"]))
        SAVE_TYPE = config["DEFAULT"]["saveformat"]
        SHOW = config["DEFAULT"].getboolean("show")
        BOUNDING_BOX = int(config["DEFAULT"]["boundingbox"])
        OUTPUT_PATH = config["DEFAULT"]["output"]
        MODEL = config["DEFAULT"]["model"]
    else:
        MODEL = arguments.model
        TRANSLATION_RANGE = arguments.translation
        ROTATION_RANGE = math.radians(arguments.rotation)
        SAMPLE_QUANTITY = arguments.samples
        BOUNDING_BOX = arguments.boundingbox
        OUTPUT_PATH = arguments.output
        IMAGE_SIZE = (arguments.resolution, arguments.resolution)
        SAVE_TYPE = arguments.saveformat
        SHOW = arguments.show

    PRELOAD = arguments.preload
    DEBUG = arguments.debug
    RANDOM_SAMPLING = arguments.random_sampling
    SPECULAR = arguments.specular
    SHADER_PATH = arguments.shader
    CAMERA_PATH = arguments.camera
    DEPTHONLY = arguments.depthonly
    SPHERE_MIN_RADIUS = arguments.minradius
    SPHERE_MAX_RADIUS = arguments.maxradius
        
    # if SHOW:
    #     import cv2
    import cv2
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    
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

    camera = Camera.load_from_json(CAMERA_PATH)
    dataset = DeepTrackLoaderBase(OUTPUT_PATH, read_data=PRELOAD)
    dataset.set_save_type(SAVE_TYPE)
    dataset.camera = camera
    window_size = (camera.width, camera.height)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    preload_count = 0
    print("Compute Mean bounding box")
    widths = []
    # for model in tqdm(MODELS):
    #     geometry_path = os.path.join(model["path"], "geometry.ply")
    #     model_3d = PlyParser(geometry_path).get_vertex()
    #     object_max_width = maximum_width(model_3d) * 1000
    #     with_add = BOUNDING_BOX / 100 * object_max_width
    #     widths.append(int(object_max_width + with_add))

    geometry_path = MODEL
    model_3d = PlyParser(geometry_path).get_vertex()
    object_max_width = maximum_width(model_3d) * 1000
    with_add = BOUNDING_BOX / 100 * object_max_width
    OBJECT_WIDTH = int(object_max_width + with_add)

    # widths.sort()
    # OBJECT_WIDTH = widths[int(len(widths)/2)]
    #OBJECT_WIDTH = 220

    metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    print("Mean object width : {}".format(OBJECT_WIDTH))

    # Iterate over all models from config files
    # for model in MODELS:
    # geometry_path = os.path.join(model["path"], "geometry.ply")
    # ao_path = os.path.join(model["path"], "ao.ply")

    geometry_path = MODEL
    ao_path = ''
    # WARNING : OBJECT_WIDTH is not the max width with multiple objects
    vpRender = ModelRenderer2(geometry_path, SHADER_PATH, dataset.camera, [window_size, IMAGE_SIZE], object_max_width=OBJECT_WIDTH)
    if os.path.exists(ao_path):
        vpRender.load_ambiant_occlusion_map(ao_path)
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
