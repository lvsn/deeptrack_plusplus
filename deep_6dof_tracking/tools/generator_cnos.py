from deep_6dof_tracking.utils.transform import Transform
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
from deep_6dof_tracking.utils.geodesic_grid import GeodesicGrid
from deep_6dof_tracking.eccv.eval_functions import get_pose_difference, compute_pose_diff
from tqdm import tqdm
import argparse
import configparser
import shutil
import os
import math
import numpy as np
import trimesh
from trimesh import Geometry
from deep_6dof_tracking.data.model_rend_test import ModelRenderer3
import open3d as o3d

ESCAPE_KEY = 1048603

import matplotlib.pyplot as plt
from typing import List


def depth_to_xyz(depth_image, fx, fy, cx, cy):
    """
    Convert a depth image to XYZ coordinates.
    
    Args:
        depth_image (numpy.ndarray): 2D depth image (H, W).
        fx, fy (float): Focal lengths in pixels.
        cx, cy (float): Principal point (optical center) in pixels.
    
    Returns:
        numpy.ndarray: 3D point cloud as an array of shape (H, W, 3).
    """
    # Get image dimensions
    height, width = depth_image.shape
    
    # Create grid of pixel coordinates
    u = np.arange(width)  # [0, 1, ..., W-1]
    v = np.arange(height) # [0, 1, ..., H-1]
    u, v = np.meshgrid(u, v)  # (H, W)
    
    # Compute X, Y, Z
    Z = depth_image  # Depth values
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack X, Y, Z into a 3D point cloud
    xyz = np.stack((X, Y, Z), axis=-1)  # Shape: (H, W, 3)
    # xyz = xyz
    xyz[:,:,-1] = xyz[:,:,-1] - 1000
    xyz[xyz[:, :, -1] < -999] = [-1, -1, -1]

    return xyz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('-c', '--camera', help="camera json path", action="store",
                        default="../data/sensors/camera_parameter_files/synthetic.json")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="model file path", action="store", default="./model_config.yaml")
    parser.add_argument('--models', help="model file path", action="store", default="./model_config.yaml")

    parser.add_argument('-y', '--saveformat', help="save format", action="store", default="numpy")
    parser.add_argument('-e', '--resolution', help="image pixel size", action="store", default=150, type=int)
    parser.add_argument('--boundingbox', help="Crop bounding box width in %% of the object max width",
                        action="store", default=10, type=int)

    parser.add_argument('-p', '--preload', help="Load any data saved in output directory", action="store_true")
    parser.add_argument('-v', '--show', help="show image while generating", action="store_true")
    parser.add_argument('-d', '--debug', help="show debug screen while generating", action="store_true")
    parser.add_argument('--specular', help="Will add random specularity to the training set", action="store_true")
    parser.add_argument('--depthonly', help="Only generate depth data", action="store_true")
    parser.add_argument('--rgbonly', help="Only generate rgb data", action="store_true")

    parser.add_argument('--config', help="config file path", action="store", default=None)

    arguments = parser.parse_args()

    

    if arguments.config is None:
        IMAGE_SIZE = (arguments.resolution, arguments.resolution)
        PRELOAD = arguments.preload
        SAVE_TYPE = arguments.saveformat
        SHOW = arguments.show
        DEBUG = arguments.debug
        SPECULAR = arguments.specular

        SHADER_PATH = arguments.shader
        OUTPUT_PATH = arguments.output
        CAMERA_PATH = arguments.camera
        BOUNDING_BOX = arguments.boundingbox
        DEPTHONLY = arguments.depthonly
        RGBONLY = arguments.rgbonly
        MODEL_PATH = arguments.model
        MODELS = yaml_load(arguments.model)["models"]
        MODELS_PATH = arguments.models
        CAMERA_PATH = arguments.camera
        SHADER_PATH = arguments.shader

    else:
        config = configparser.ConfigParser()
        config.read(arguments.config)
        res = int(config['DEFAULT']['resolution'])
        IMAGE_SIZE = (res, res)
        PRELOAD = config['DEFAULT'].getboolean('preload')
        SAVE_TYPE = config['DEFAULT']['saveformat']
        SHOW = config['DEFAULT'].getboolean('show')
        DEBUG = config['DEFAULT'].getboolean('debug')
        SPECULAR = config['DEFAULT'].getboolean('specular')

        OUTPUT_PATH = config['DEFAULT']['output']
        BOUNDING_BOX = int(config['DEFAULT']['boundingbox'])
        DEPTHONLY = config['DEFAULT'].getboolean('depthonly')
        RGBONLY = config['DEFAULT'].getboolean('rgbonly')
        MODEL_PATH = config['DEFAULT']['model']
        MODELS = yaml_load(config['DEFAULT']['model'])["models"]
        MODELS_PATH = config['DEFAULT']['models']
        CAMERA_PATH = config['DEFAULT']['camera']
        SHADER_PATH = config['DEFAULT']['shader']

    # if SHOW:
    #     import cv2
    import cv2

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    if MODELS_PATH is None:
        shutil.copy(MODEL_PATH, os.path.join(OUTPUT_PATH, "models.yml"))
    else:
        MODELS = []
        objects = os.listdir(MODELS_PATH)
        for obj in objects:
            if os.path.isdir(os.path.join(MODELS_PATH, obj)):
                MODELS.append({"path": os.path.join(MODELS_PATH, obj),
                               "name": obj})
            


    # Write important misc data to file
    metadata = {}
    metadata["image_size"] = str(IMAGE_SIZE[0])
    metadata["save_type"] = SAVE_TYPE
    metadata["object_width"] = {}
    metadata["bounding_box"] = BOUNDING_BOX

    camera = Camera.load_from_json(CAMERA_PATH)
    IMAGE_SIZE = (camera.width, camera.height)

    window_size = (camera.width, camera.height)
    preload_count = 0
    print("Compute Mean bounding box")
    widths = []
    
    use_ply = True
    rescale=False
    pad_name=True

    if use_ply:
        for model in tqdm(MODELS):
            ply_files = [f for f in os.listdir(model["path"]) if f.endswith(".ply")]
            if len(ply_files) == 0:
                continue
            ply_file = ply_files[0]
            geometry_path = os.path.join(model["path"], ply_file)
            model_3d = PlyParser(geometry_path).get_vertex()
            object_max_width = maximum_width(model_3d) * 1000
            with_add = BOUNDING_BOX / 100 * object_max_width
            if int(object_max_width + with_add) > 10000:
                widths.append(int((object_max_width + with_add)/1000))
                rescale = True
            else:
                widths.append(int(object_max_width + with_add))
    else:
        for model in tqdm(MODELS):
            files = os.listdir(model["path"])
            print(model["name"])
            for file in files:
                if file.endswith(".obj"):
                    print(file)
                    print(file)
                    geometry_path = os.path.join(model["path"], file)
                    mesh : Geometry = trimesh.load(geometry_path)
                    model_3d = mesh.vertices
                    object_max_width = maximum_width(model_3d) * 1000
                    # metadata[model]["object_width_individual"] = str(object_max_width)
                    # true_widths[model] = object_max_width
                    with_add = BOUNDING_BOX / 100 * object_max_width
                    if int(object_max_width + with_add) > 10000:
                        widths.append(int((object_max_width + with_add)/1000))
                        rescale = True
                    else:
                        widths.append(int(object_max_width + with_add))
                    # widths[model] = object_max_width + with_add

    widths.sort()
    OBJECT_WIDTH = widths[int(len(widths)/2)]
    #OBJECT_WIDTH = 220
    print(len(MODELS))
    metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    print("Mean object width : {}".format(OBJECT_WIDTH))
    # Iterate over all models from config files

    grid = GeodesicGrid()
    grid.refine_icoshpere(1)
    v = grid.cloud.vertex['XYZ']
    v = v[~np.all(v == [0,0,0], axis=1)]
    samples : List[Transform] = []
    for i in v:
        if i[0] == 0 and i[1] == 0:
            i[0] += 0.1
            i[1] += 0.1
        view = Transform.lookAt(i, np.zeros(3), np.array([0, 0, 1]))
        samples.append(view)

    for model in MODELS:
        ply_files = [f for f in os.listdir(model["path"]) if f.endswith(".ply")]
        if len(ply_files) == 0:
            continue
        
        dataset = DeepTrackLoaderBase(os.path.join(OUTPUT_PATH, model["name"]), read_data=PRELOAD)
        dataset.set_save_type(SAVE_TYPE)
        dataset.camera = camera
        
        print(model["name"])
        if use_ply:
            ply_file = [f for f in os.listdir(model["path"]) if f.endswith(".ply")][0]
            geometry_path = os.path.join(model["path"], ply_file)
            ao_path = os.path.join(model["path"], "ao.ply")
            vpRender = ModelRenderer2(geometry_path, SHADER_PATH, dataset.camera, [window_size, IMAGE_SIZE], object_max_width=OBJECT_WIDTH, model_scale=0.001 if rescale else 1)
            if os.path.exists(ao_path):
                vpRender.load_ambiant_occlusion_map(ao_path)
        else:
            files = os.listdir(model["path"])
            for file in files:
                if file.endswith(".obj"):
                    geometry_path = os.path.join(model["path"], file)
                elif file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    texture_path = os.path.join(model["path"], file)
            vpRender = ModelRenderer3(geometry_path,
                                      SHADER_PATH, 
                                      texture_path,
                                      dataset.camera,
                                      [window_size, IMAGE_SIZE],
                                      original_width=OBJECT_WIDTH,
                                      new_width=OBJECT_WIDTH)
            
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"]), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], 'xyz'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], 'depth'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], 'masks'), exist_ok=True)
        
        for i, s in enumerate(samples):
            s_mat = s.matrix
            R_x = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, 1, 0],
                [0,  0,  0, 1]
            ])
            new_mat = R_x @ s_mat
            s_render = Transform.from_matrix(new_mat)
            # print(s_render)

            bb = compute_2Dboundingbox(s_render, dataset.camera, OBJECT_WIDTH, scale=(1000, 1000, -1000))
            left = np.min(bb[:, 1])
            right = np.max(bb[:, 1])
            top = np.min(bb[:, 0])
            bottom = np.max(bb[:, 0])
            # vpRender.setup_camera(camera, left, right, bottom, top)
            vpRender.setup_camera(camera, 0, camera.width, 0, camera.height)
            rgbA, depthA = vpRender.render_image(s_render, fbo_index=1)
            maskA = np.zeros_like(rgbA)
            maskA[rgbA!=0] = 255

            index = dataset.add_pose(rgbA, depthA, s, maskA, depthonly=DEPTHONLY)
            xyz = depth_to_xyz(depthA, camera.focal_x, camera.focal_y, camera.center_x, camera.center_y)
            np.save(os.path.join(OUTPUT_PATH, model["name"], 'xyz', '{:05d}.npy'.format(int(i))), xyz)

            if i % 500 == 0:
                dataset.dump_images_on_disk(pad_name=pad_name)
            if i % 5000 == 0:
                dataset.save_json_files(metadata)

            if DEBUG:
                show_frames(rgbA, depthA, np.zeros_like(rgbA), np.zeros_like(depthA))
            if SHOW:
                cv2.imshow("test", np.concatenate((rgbA[:, :, ::-1], np.zeros_like(rgbA)[:, :, ::-1]), axis=1))
                k = cv2.waitKey(1)
                if k == ESCAPE_KEY:
                    break

    

        dataset.dump_images_on_disk(pad_name=pad_name)
        dataset.save_json_files(metadata)

        # if RGBONLY:
        #     out_files = os.listdir(os.path.join(OUTPUT_PATH, model["name"]))
        #     for f in out_files:
        #         if "m" in f or "d" in f:
        #             os.remove(os.path.join(OUTPUT_PATH, model["name"], f))