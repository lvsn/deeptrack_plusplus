import json

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.modelrenderer import InitOpenGL, ModelRenderer

import cv2
import os
import time
import numpy as np

from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.data.utils import image_blend

ESCAPE_KEY = 1048603
SPACE_KEY = 1048608

if __name__ == '__main__':
    """
    Check if all detections in sequence are as expected
    """

    save_video = True


    # Populate important data from config file
    object_name = "shoe"
    experimentation = "fix_occluded_1"

    SEQUENCE_PATH = "/media/ssd/eccv/sequences/{}_{}".format(object_name, experimentation)
    MODEL_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/"
    MODEL_GEO_PATH = os.path.join(MODEL_PATH, object_name, "geometry.ply")
    MODEL_AO_PATH = os.path.join(MODEL_PATH, object_name, "ao.ply")
    SHADER_PATH = "../data/shaders"
    fps = 30
    dataset = DeepTrackLoaderBase(SEQUENCE_PATH)
    view_camera_size = (int(dataset.camera.width/1.5), int(dataset.camera.height/1.5))

    vpRender = ModelRenderer2(MODEL_GEO_PATH, SHADER_PATH, dataset.camera, [view_camera_size])
    vpRender.load_ambiant_occlusion_map(MODEL_AO_PATH)


    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(os.path.join("/media/ssd/eccv/", "video.avi"), fourcc, 24.0, view_camera_size)

    print("Sequence length: {}".format(len(dataset.data_pose)))
    for i, (frame, pose) in enumerate(dataset.data_pose):
        time_start = time.time()
        markers = json.load(open(os.path.join(SEQUENCE_PATH, "{}.json".format(i))))
        rgb, depth = frame.get_rgb_depth(SEQUENCE_PATH)

        rgb = cv2.resize(rgb, view_camera_size)
        depth = cv2.resize(depth, view_camera_size)

        rgb_render, depth_render = vpRender.render_image(pose)
        blend = image_blend(rgb_render[:, :, ::-1], rgb)

        cv2.imshow("debug", blend[:, :, ::-1])
        cv2.imshow("debug_depth", (depth / np.max(depth) * 255).astype(np.uint8))

        if save_video:
            video_out.write(blend[:, :, ::-1])

        elapsed_time = time.time() - time_start
        wait = elapsed_time - (1./fps)
        cv2.waitKey(int(max(1., wait*1000)))

    if save_video:
        video_out.release()

