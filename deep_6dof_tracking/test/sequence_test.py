from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.modelrenderer import InitOpenGL, ModelRenderer

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

from deep_6dof_tracking.data.sequence_loader import SequenceLoader
from deep_6dof_tracking.data.utils import image_blend, compute_2Dboundingbox, normalize_scale

ESCAPE_KEY = 1048603
SPACE_KEY = 1048608


def compute_render(renderer, camera, pose, bb):
    left = np.min(bb[:, 1])
    right = np.max(bb[:, 1])
    top = np.min(bb[:, 0])
    bottom = np.max(bb[:, 0])
    renderer.setup_camera(camera, left, right, bottom, top)
    render_rgb, render_depth = renderer.render_image(pose)
    return render_rgb, render_depth

if __name__ == '__main__':
    """
    Check if all detections in sequence are as expected
    """

    # Populate important data from config file
    object_name = "dragon"
    experimentation = "motion_hard"

    CROPPED = True

    SEQUENCE_PATH = "/media/ssd/eccv/Sequences/final_sequences/{}_{}".format(object_name, experimentation)
    #SEQUENCE_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/deeptracking_eccv/sequences/{}_{}".format(object_name, experimentation)
    MODEL_GEO_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/{}/geometry.ply".format(object_name)
    MODEL_AO_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/{}/ao.ply".format(object_name)
    SHADER_PATH = "../data/shaders"
    #OUTPUT_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/to_delete"
    fps = 1
    dataset = SequenceLoader(SEQUENCE_PATH)
    window_size = (dataset.camera.width, dataset.camera.height)
    if CROPPED:
        window_size = (174, 174)
    steps = 30
    window = InitOpenGL(*window_size)
    vpRender = ModelRenderer(MODEL_GEO_PATH, SHADER_PATH, dataset.camera, window, window_size)
    vpRender.load_ambiant_occlusion_map(MODEL_AO_PATH)
    print("Sequence length: {}".format(len(dataset.data_pose)))
    for i, (frame, pose) in enumerate(dataset.data_pose):
        time_start = time.time()
        rgb, depth = frame.get_rgb_depth(SEQUENCE_PATH)
        depth -= 14
        bb2 = compute_2Dboundingbox(pose, dataset.camera, 150, scale=(1000, -1000, -1000))
        rgbB, depthB = normalize_scale(rgb, depth, bb2, window_size)

        if CROPPED:
            bb = compute_2Dboundingbox(pose, dataset.camera, 150, scale=(1000, 1000, -1000))
            rgbA, depthA = compute_render(vpRender, dataset.camera, pose, bb)
            depthA = depthA.astype(float)
            depthB = depthB.astype(float)
            plt.subplot("131")
            plt.imshow(depthA, vmin=1000, vmax=np.max(depthA))
            plt.subplot("132")
            plt.imshow(depthB, vmin=1000, vmax=np.max(depthA))
            plt.subplot("133")
            plt.imshow(depthA - depthB, vmin=-40, vmax=40)
            if i % steps == 0:
                plt.show()

        else:
            rgb_render, depth_render = vpRender.render_image(pose)
            blend = image_blend(rgb_render[:, :, ::-1], rgb)
            cv2.imshow("debug", blend[:, :, ::-1])
            elapsed_time = time.time() - time_start
            wait = elapsed_time - (1./fps)
            cv2.waitKey()


