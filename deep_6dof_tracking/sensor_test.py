
import sys
import time
import numpy as np
import os
from tqdm import tqdm
from importlib._bootstrap_external import SourceFileLoader

from deep_6dof_tracking.utils.camera import Camera
from pytorch_toolbox.io import yaml_load
from py_rgbd_grabber.kinect2 import Kinect2
from deep_6dof_tracking.data.utils import compute_axis, image_blend
from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch
from deep_6dof_tracking.image_show import ImageShow, ImageShowMessage, ESCAPE_KEY
from deep_6dof_tracking.utils.transform import Transform
import cv2


def draw_debug(img, pose, gt_pose, tracker, alpha, debug_info):
    if debug_info is not None:
        img_render, bb, _ = debug_info
        img_render = cv2.resize(img_render, (bb[2, 1] - bb[0, 1], bb[1, 0] - bb[0, 0]))
        crop = img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :]
        h, w, c = crop.shape
        blend = image_blend(img_render[:h, :w, ::-1], crop)
        img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :] = cv2.addWeighted(img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :],
                                                                       1 - alpha, blend, alpha, 1)
    else:
        axis = compute_axis(pose, tracker.camera)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, 255), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, 255, 0), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (255, 0, 0), 3)
        if gt_pose is not None:
            axis_gt = compute_axis(gt_pose, tracker.camera)
            cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[1, ::-1]), (0, 0, 155), 3)
            cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[2, ::-1]), (0, 155, 0), 3)
            cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[3, ::-1]), (155, 0, 0), 3)

alpha = 1
def trackbar(x):
    global alpha
    alpha = x/100


def preprocess_function(frame):
    frame.rgb = cv2.resize(frame.rgb, (camera.width, camera.height))
    frame.depth = cv2.resize(frame.depth, (camera.width, camera.height))
    return frame

if __name__ == '__main__':

    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "sensor_config.yml"
    configs = yaml_load(config_path)

    # Populate important data from config file
    OUTPUT_PATH = configs["output_path"]
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    NETWORK_PATH = configs["network_path"]
    SHADER_PATH = configs["shader_path"]
    ITERATION = configs["iteration"]
    SAVE_VIDEO = configs["save_video"]
    SHOW_DEPTH = configs["show_depth"]
    SHOW_ZOOM = configs["show_zoom"]
    CAMERA_PATH = configs["camera_path"]
    DEBUG = configs["debug"]
    DEBUG_TIME = configs["debug_time"]
    DEBUG_SUB_TIME = configs["debug_sub_time"]
    BACKEND = configs["backend"]
    MODEL_3D_PATH_GEO = configs["geometry_path"]
    NEW_RENDERER = configs["new_renderer"]

    camera = Camera.load_from_json(CAMERA_PATH)
    sensor = Kinect2()
    sensor.camera = camera

    #show_screen = ImageShow(SHOW_ZOOM, os.path.join(OUTPUT_PATH, "video.avi"), 15, camera.width, camera.height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(os.path.join(OUTPUT_PATH, "video.avi"), fourcc, 17,
                                   (int(camera.width / 2), int(camera.height / 2)))
    frame_buffer = []
    initial_pose = Transform.from_parameters(0, 0, -1, 0, 0, 0)
    set_initial_pose = True

    out = None
    debug_info = None
    total_start_time = time.time()
    network_module = SourceFileLoader("network", os.path.join(NETWORK_PATH, "network.py")).load_module()
    tracker = DeepTrackerBatch(camera, BACKEND, network_module.Network, new_renderer=NEW_RENDERER)
    tracker.load(os.path.join(NETWORK_PATH, "model_best.pth.tar"), None, MODEL_3D_PATH_GEO, None, SHADER_PATH)

    sensor.initialize_()
    while True:
        rgbd_frame = sensor.get_frame_()

        if DEBUG_TIME:
            start_time = time.time()

        current_rgb = rgbd_frame.rgb
        current_depth = rgbd_frame.depth
        screen = current_rgb

        if SHOW_DEPTH:
            screen_depth = (current_depth / np.max(current_depth) * 255).astype(np.uint8)[:, :, np.newaxis]
            screen = np.repeat(screen_depth, 3, axis=2)

        if set_initial_pose:
            previous_pose = initial_pose
        else:
            for j in range(ITERATION):
                time_beg = time.time()
                predicted_pose, debug_info = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth,
                                                                           verbose=DEBUG,
                                                                           debug_time=DEBUG_SUB_TIME,
                                                                           batch=False,
                                                                           iteration=j)
                previous_pose = predicted_pose

        time_beg = time.time()
        draw_debug(screen, previous_pose, None, tracker, alpha, debug_info)

        screen = cv2.resize(screen, (int(screen.shape[1] / 2), int(screen.shape[0] / 2)))
        if SHOW_ZOOM and debug_info is not None:
            _, _, zoom = debug_info
            zoom_h, zoom_w, zoom_c = zoom.shape
            screen[:zoom_h + 6, :zoom_w + 6, :] = 255
            screen[3:zoom_h + 3, 3:zoom_w + 3, :] = zoom
        cv2.imshow("test", screen[:, :, ::-1])
        if SAVE_VIDEO:
            frame_buffer.append(screen[:, :, ::-1])
        key = cv2.waitKey(1)
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key_chr == " ":
            set_initial_pose = not set_initial_pose
        if key == ESCAPE_KEY:
            break

        if DEBUG_TIME:
            stop_time = time.time()
            compute_time = stop_time - start_time
            total_time = stop_time - total_start_time
            total_start_time = time.time()
            fps = 1/compute_time
            print("FPS:{:0.4f} Estimation processing time : {}".format(1/total_time, compute_time))
    sensor.clean_()
    for frame in tqdm(frame_buffer):
        video.write(frame)
    video.release()

