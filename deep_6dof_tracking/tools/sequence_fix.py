"""
    use a pose detector (aruco, checkboard) and compute the pose on the whole dataset
"""
import sys
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.utils import image_blend
from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
import cv2
import os
import numpy as np


if __name__ == '__main__':
    """
    Simple tool to remove frames
    """
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "sequence_fix_config.yml"
    configs = yaml_load(config_path)

    dataset_path = configs["dataset_path"]
    output_path = configs["output_path"]
    model_path = configs["model_path"]
    model_ao_path = configs["model_ao_path"]
    shader_path = configs["shader_path"]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    dataset_in = DeepTrackLoader(dataset_path)
    dataset_out = DeepTrackLoader(output_path)
    dataset_out.set_save_type(dataset_in.metadata["save_type"])
    dataset_out.camera = dataset_in.camera
    window = InitOpenGL(dataset_in.camera.width, dataset_in.camera.height)
    vpRender = ModelRenderer(model_path, shader_path, dataset_in.camera, window, (dataset_in.camera.width, dataset_in.camera.height))
    vpRender.load_ambiant_occlusion_map(model_ao_path)
    ground_truth_pose = None

    for i, (frame, pose) in enumerate(dataset_in.data_pose):
        rgb, depth = frame.get_rgb_depth(dataset_in.root)
        rgb_render, depth_render = vpRender.render_image(pose)
        blend = image_blend(rgb_render[:, :, ::-1], rgb)
        cv2.imshow("debug", blend[:, :, ::-1])
        key = cv2.waitKey()
        key_chr = chr(key & 255)
        if key_chr == "r":
            # Reject frame
            print("Frame {} rejected".format(i))
        else:
            dataset_out.add_pose(rgb, depth, pose)


    dataset_out.dump_images_on_disk()
    dataset_out.save_json_files(dataset_in.metadata)
