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

from deep_6dof_tracking.utils.aruco import ArucoDetector

if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "detection_config.yml"
    configs = yaml_load(config_path)

    dataset_path = configs["dataset_path"]
    detector_path = configs["detector_path"]
    model_path = configs["model_path"]
    model_ao_path = configs["model_ao_path"]
    shader_path = configs["shader_path"]

    os.remove(os.path.join(dataset_path, "viewpoints.json"))
    dataset = DeepTrackLoader(dataset_path)
    dataset.set_save_type("png")
    offset = Transform.from_matrix(np.load(os.path.join(dataset.root, "board2object.npy")))

    camera = Camera.load_from_json(dataset_path)
    dataset.camera = camera
    files = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[-1] == ".png" and 'd' not in os.path.splitext(f)[0]]
    detector = ArucoDetector(camera, detector_path)
    window = InitOpenGL(camera.width, camera.height)
    vpRender = ModelRenderer(model_path, shader_path, camera, window, (camera.width, camera.height))
    vpRender.load_ambiant_occlusion_map(model_ao_path)
    ground_truth_pose = None

    for i in range(len(files)):
        img = cv2.imread(os.path.join(dataset.root, "{}.png".format(i)))
        detection = detector.detect(img)
        if detection is not None:
            ground_truth_pose = detection
            ground_truth_pose.combine(offset.inverse(), copy=False)
        else:
            print("[WARN]: frame {} has not been detected.. using previous detection".format(i))
        dataset.add_pose(None, None, ground_truth_pose)
        rgb_render, depth_render = vpRender.render_image(ground_truth_pose)
        bgr_render = rgb_render.copy()
        img = image_blend(bgr_render, img)

        cv2.imshow("view", img)
        cv2.waitKey(1)

    dataset.save_json_files({"save_type": "png"})
