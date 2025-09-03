"""
    use a pose detector (aruco, checkboard) and compute the pose on the whole dataset
"""
import sys
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.utils import image_blend
from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deep_6dof_tracking.utils.angles import mat2euler, euler2mat
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
import cv2
import os
import numpy as np


if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "detection_config.yml"
    configs = yaml_load(config_path)

    dataset_path = configs["dataset_path"]
    model_path = configs["model_path"]
    model_ao_path = configs["model_ao_path"]
    shader_path = configs["shader_path"]

    dataset = DeepTrackLoader(dataset_path)

    camera = Camera.load_from_json(dataset_path)
    dataset.camera = camera
    window = InitOpenGL(camera.width, camera.height)
    vpRender = ModelRenderer(model_path, shader_path, camera, window, (camera.width, camera.height))
    vpRender.load_ambiant_occlusion_map(model_ao_path)
    ground_truth_pose = None

    for i in range(dataset.size()):
        img = cv2.imread(os.path.join(dataset.root, "{}.png".format(i)))
        frame, ground_truth_pose = dataset.data_pose[i]
        params = ground_truth_pose.to_parameters()
        ground_truth_pose.set_translation(params[0], params[1], params[2])
        ground_truth_pose.matrix[0:3, 0:3] = euler2mat(params[3], params[4], params[5])

        rgb_render, depth_render = vpRender.render_image(ground_truth_pose)
        bgr_render = rgb_render.copy()
        img = image_blend(bgr_render, img)

        cv2.imshow("view", img)
        cv2.waitKey(1)
    print("Save viewpoint.json")
    dataset.save_json_files(dataset.metadata)