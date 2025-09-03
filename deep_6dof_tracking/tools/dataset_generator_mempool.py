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
ESCAPE_KEY = 1048603

import matplotlib.pyplot as plt
from typing import List


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('-c', '--camera', help="camera json path", action="store",
                        default="../data/sensors/camera_parameter_files/synthetic.json")
    parser.add_argument('--shader', help="shader path", action="store", default="../data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="model file path", action="store", default="./model_config.yaml")

    parser.add_argument('-y', '--saveformat', help="save format", action="store", default="numpy")
    parser.add_argument('-e', '--resolution', help="image pixel size", action="store", default=150, type=int)
    parser.add_argument('--boundingbox', help="Crop bounding box width in %% of the object max width",
                        action="store", default=10, type=int)

    parser.add_argument('-p', '--preload', help="Load any data saved in output directory", action="store_true")
    parser.add_argument('-v', '--show', help="show image while generating", action="store_true")
    parser.add_argument('-d', '--debug', help="show debug screen while generating", action="store_true")
    parser.add_argument('--specular', help="Will add random specularity to the training set", action="store_true")
    parser.add_argument('--depthonly', help="Only generate depth data", action="store_true")

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
        MODEL_PATH = arguments.model
        MODELS = yaml_load(arguments.model)["models"]
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
            
    pad_name = True

    # Write important misc data to file
    metadata = {}
    metadata["image_size"] = str(IMAGE_SIZE[0])
    metadata["save_type"] = SAVE_TYPE
    metadata["object_width"] = {}
    metadata["bounding_box"] = BOUNDING_BOX

    camera = Camera.load_from_json(CAMERA_PATH)
    IMAGE_SIZE = (camera.width, camera.height)

    # dataset = DeepTrackLoaderBase(OUTPUT_PATH, read_data=PRELOAD)
    # dataset.set_save_type(SAVE_TYPE)
    # dataset.camera = camera
    window_size = (camera.width, camera.height)
    preload_count = 0
    print("Compute Mean bounding box")
    widths = []


    # # Commenetd for now
    # for model in tqdm(MODELS):
    #     geometry_path = os.path.join(model["path"], "geometry.ply")
    #     model_3d = PlyParser(geometry_path).get_vertex()
    #     object_max_width = maximum_width(model_3d) * 1000
    #     with_add = BOUNDING_BOX / 100 * object_max_width
    #     widths.append(int(object_max_width + with_add))
    # widths.sort()
    # OBJECT_WIDTH = widths[int(len(widths)/2)]


    OBJECT_WIDTH = 250
    
    
    print(len(MODELS))
    metadata["bounding_box_width"] = str(OBJECT_WIDTH)
    print("Mean object width : {}".format(OBJECT_WIDTH))
    # Iterate over all models from config files

    grid = GeodesicGrid()
    grid.refine_icoshpere(2)
    v = grid.cloud.vertex['XYZ']
    v = v[~np.all(v == [0,0,0], axis=1)]
    samples : List[Transform] = []
    for i in v:
        if i[0] == 0 and i[1] == 0:
            i[0] += 0.1
            i[1] += 0.1
        view = Transform.lookAt(i, np.zeros(3), np.array([0, 0, 1]))
        samples.append(view)

    # samples = []
    # names = []
    # path = 'C:\\Users\\renau\\Documents\\Uni\\maitrise\\source\\BundleSDF\\data\\out\\clock_interaction_hard_small\\00002\\selected'
    # for f in os.listdir(path):
    #     if 'pose' in f:
    #         pose = np.loadtxt(os.path.join(path, f))
    #         samples.append(Transform.from_matrix(pose))

    # new_mat = np.array([[ 0.90478027, -0.3555298,   0.23449591, -0.07626432],
    #                     [-0.04903804,  0.45997036,  0.8865871,   0.25357303],
    #                     [-0.4230669,  -0.8136604,   0.39873332, -1.086932  ],
    #                     [ 0,          0,          0,          1,        ]])
    # # R_x = np.array([
    # #     [1,  0,  0, 0],
    # #     [0, -1,  0, 0],
    # #     [0,  0, 1, 0],
    # #     [0,  0,  0, 1]
    # # ])
    # # # Flip upside down
    # # new_mat = R_x @ new_mat
    # # new_pose = Transform.from_matrix(new_mat)
    # # print(new_pose.to_parameters())
    # # print(new_pose.matrix)
    # # samples.append(new_pose)
    # samples.append(Transform.from_matrix(new_mat))

    # def gt_ref_2_fp_ref(pose):
    #     pose[0,0] = -pose[0,0]
    #     pose[0,1] = -pose[0,1]
    #     pose[1,2] = -pose[1,2]
    #     pose[2,2] = -pose[2,2]
    #     pose[1,3] = -pose[1,3]
    #     pose[2,3] = -pose[2,3]
    #     return pose

    # samples : List[Transform] = []
    # # Test for the rotation thing
    # mp2 = np.array([[-8.5065085e-01, -7.9775324e-09, -5.2573115e-01,  5.2573103e-01],
    #                 [ 5.2573115e-01, -2.8082091e-08, -8.5065085e-01,  8.5065079e-01],
    #                 [-7.9775333e-09, -1.0000000e+00,  2.8082093e-08, -2.8082090e-08],
    #                 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
    # mp95 = np.array([[-7.1128178e-01,  1.1290245e-01, -6.9378048e-01,  6.9378042e-01],
    #                 [ 7.0290703e-01,  1.1424765e-01, -7.0204645e-01,  7.0204639e-01],
    #                 [ 2.6632451e-08, -9.8701596e-01, -1.6062219e-01,  1.6062218e-01],
    #                 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
    
    # mp2 = np.linalg.inv(mp2)
    # mp95 = np.linalg.inv(mp95)
    # mp2 = gt_ref_2_fp_ref(mp2)
    # mp95 = gt_ref_2_fp_ref(mp95)

    # print(mp2)
    # print(mp95)
    # samples.append(Transform.from_matrix(mp2))
    # samples.append(Transform.from_matrix(mp95))
    # r = -6.988101436076153
    # R_z2 = np.array([
    #             [np.cos(np.radians(r)), -np.sin(np.radians(r)), 0, 0],
    #             [np.sin(np.radians(r)), np.cos(np.radians(r)), 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1]
    #         ])
    # r2 = R_z2 @ mp2
    # samples.append(Transform.from_matrix(r2))
    # r = -20.9772099506652
    # R_z95 = np.array([
    #             [np.cos(np.radians(r)), -np.sin(np.radians(r)), 0, 0],
    #             [np.sin(np.radians(r)), np.cos(np.radians(r)), 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1]
    #         ])
    # r95 = R_z95 @ mp2
    # samples.append(Transform.from_matrix(r95))
    



    # min_ts = np.ones(len(samples))*np.inf
    # min_rs = np.ones(len(samples))*np.inf
    # indices = np.ones(len(samples))

    # from scipy.spatial.transform import Rotation as R

    # def relative_rotation(R1, R2):
    #     # Compute relative rotation matrix
    #     R_rel = R1.T @ R2

    #     # Check for 180-degree rotation
    #     trace = np.trace(R_rel)
    #     if np.isclose(trace, -1):
    #         # Handle 180-degree rotation
    #         axis = np.array([
    #             R_rel[2, 1] - R_rel[1, 2],
    #             R_rel[0, 2] - R_rel[2, 0],
    #             R_rel[1, 0] - R_rel[0, 1]
    #         ])
    #         axis = axis / np.linalg.norm(axis)
    #         return 180, axis
    #     else:
    #         # Use scipy to handle general rotations
    #         r = R.from_matrix(R_rel)
    #         angle = np.degrees(r.magnitude())
    #         axis = r.as_rotvec() / np.linalg.norm(r.as_rotvec())
    #         return angle, axis

    # for i in range(len(samples)):
    #     for j in range(len(samples)):
    #         if i == j:
    #             continue
    #         if i==63 and j==4:
    #             print('here')
    #         # Md = np.linalg.inv(samples[j].matrix) @ samples[i].matrix
    #         # R = Md[:3, :3]
    #         # R4 = np.eye(4)
    #         # R4[:3, :3] = R
    #         # trans = Transform.from_matrix(R4)
    #         # diff = trans.to_parameters()
    #         # pose_diff = diff[np.newaxis, :]
    #         # diff_t, diff_r = compute_pose_diff(pose_diff)
    #         # diff_r = np.rad2deg(diff_r)
    #         diff_t = 0
    #         diff_r, _ = relative_rotation(samples[j].matrix[:3, :3], samples[i].matrix[:3, :3])
    #         if diff_t < min_ts[i]:
    #             min_ts[i] = diff_t
    #         if diff_r < min_rs[i]:
    #             min_rs[i] = diff_r
    #             indices[i] = j
    # np.set_printoptions(edgeitems=200, suppress=True)
    # print('min_rs')
    # print(np.sort(min_rs))
    # print(np.argsort(min_rs))
    # print(indices)

    # rot_values = np.linspace(0, 360, 9)[:-1]
    # print(rot_values)


    for model in MODELS:
        geometry_path = os.path.join(model["path"], "geometry.ply")
        ao_path = os.path.join(model["path"], "ao.ply")

        dataset = DeepTrackLoaderBase(os.path.join(OUTPUT_PATH, model["name"]), read_data=PRELOAD)
        dataset.set_save_type(SAVE_TYPE)
        dataset.camera = camera
        
        vpRender = ModelRenderer2(geometry_path, SHADER_PATH, dataset.camera, [window_size, IMAGE_SIZE], object_max_width=OBJECT_WIDTH)
        if os.path.exists(ao_path):
            vpRender.load_ambiant_occlusion_map(ao_path)

        os.makedirs(os.path.join(OUTPUT_PATH, model["name"]), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], "rgb"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], "depth"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], "masks"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, model["name"], "annotated_poses"), exist_ok=True)

        for i, s in enumerate(samples):
            # for r in rot_values:
            # for r in [0, 180]:
            # if i == 25 or i == 28:
            #     print('here')
            #     continue

            # s_rot = s.copy()
            # s_rot.rotate(0, r, 0, is_degree=True)

            # s_mat = s.matrix
            # R_z = np.array([
            #     [np.cos(np.radians(r)), -np.sin(np.radians(r)), 0, 0],
            #     [np.sin(np.radians(r)), np.cos(np.radians(r)), 0, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ])
            # new_mat = R_z @ s_mat
            # s = Transform.from_matrix(new_mat)

            # testing the rotation poses outputed by bsdf
            # s_mat = s.matrix
            # s_mat = np.linalg.inv(s_mat)
            # s_mat = gt_ref_2_fp_ref(s_mat)
            # s = Transform.from_matrix(s_mat)


            # Apply mirror transform along x-axis.
            # Weird but required to fit in the BundleSDF reference frame
            # Still need to save the ORIGINAL pose
            
            # Render the object closer to the camera
            s = s.copy().translate(0, 0, 0.5)
            s_mat = s.matrix
            R_x = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, 1, 0],
                [0,  0,  0, 1]
            ])
            new_mat = R_x @ s_mat
            s_render = Transform.from_matrix(new_mat)

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

            
            # for r2 in rot_values:
            #     plt.subplot(1,2,1)
            #     plt.imshow(rgbA)
            #     plt.axis('off')
            #     plt.subplot(1,2,2)
            #     # print(R_x)
            #     # print(R_x[:2, :3])
            #     # im2 = cv2.warpAffine(rgbA, R_x[:2, :3].astype(float), (rgbA.shape[1], rgbA.shape[0]))
            #     print(r2)
            #     center = (rgbA.shape[1] // 2, rgbA.shape[0] // 2)
            #     # Calculate the rotation matrix
            #     M = cv2.getRotationMatrix2D(center, r2, scale=1.0)
            #     im2 = cv2.warpAffine(rgbA, M, (rgbA.shape[1], rgbA.shape[0]))
            #     plt.imshow(im2)
            #     plt.axis('off')
            #     plt.show()

            index = dataset.add_pose(rgbA, depthA, s, maskA, depthonly=DEPTHONLY)
            np.savetxt(os.path.join(OUTPUT_PATH, model["name"], "annotated_poses", "{:05d}.txt".format(i)), s.matrix)

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
