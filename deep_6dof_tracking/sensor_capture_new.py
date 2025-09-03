import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.data.utils import compute_axis, image_blend, compute_2Dboundingbox
from deep_6dof_tracking.data.sensor_wrapper import SensorWrapper, SensorType
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch

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

started = False
def on_mouse_click(event, x, y, flags, param):
    global started
    if event == cv2.EVENT_LBUTTONDOWN:
        started = True

if __name__ == '__main__':
    # Loading configs
    config_path = 'deep_6dof_tracking/sensor_config.yml'
    configs = yaml_load(config_path)

    output_path = configs["output_path"]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    sequence_name = configs["sequence_name"]
    output_path = os.path.join(output_path, sequence_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    network_path = configs["network_path"]
    shader_path = configs["shader_path"]
    iterations = configs["iteration"]
    save_video = configs["save_video"]
    camera_path = configs["camera_path"]
    backend = configs["backend"]
    geometry_path = configs["geometry_path"]
    save_frames = configs["save_frames"]
    depth_only = configs["depth_only"]

    #Getting camera model
    camera = Camera.load_from_json(camera_path)

    # To render a model at initial pose before capturing
    initialize_pose = True
    initial_pose = Transform.from_parameters(0, 0, -1, -1.57, 0, 0)
    initial_pose = Transform.from_parameters(0, 0, -1, -2.355, 0, 0)
    # initial_pose = Transform.from_parameters(0, 0, -1, -1.57, 0, 0)
    from deep_6dof_tracking.networks.deeptrack_res_net import DeepTrackResNet
    tracker = DeepTrackerBatch(camera, backend, DeepTrackResNet)
    tracker.load(os.path.join(network_path, "model_best.pth.tar"), None, geometry_path, None, shader_path)
    print(tracker.object_width)
    bb = compute_2Dboundingbox(initial_pose, camera, tracker.object_width, scale=(1000, 1000, -1000))
    rgbA, depthA = tracker.compute_render(initial_pose, bb)
    debug_info = (rgbA, bb, np.hstack((rgbA, rgbA)))

    # Initializing camera
    sensor = SensorWrapper(SensorType.HELIOS2)
    sensor.initialize()
    sensor.start_stream()

    # If we want to save video/frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, 'video.mp4'), fourcc, 11.0, (640, 480))
    color_frames = []
    depth_frames = []
    heatmap_frames = []

    # Creating windows for diplay
    cv2.namedWindow('color')
    cv2.namedWindow('depth')
    cv2.namedWindow('heatmap')
    cv2.setMouseCallback('heatmap', on_mouse_click)

    # Loop
    last_time = time.time()
    DEBUG_TIME=True
    print('waiting for click')
    while True:
        img, depth_img, heatmap = sensor.get_last_RGBD_frame(heatmap=True)

        if started:
            if save_frames or save_video:
                color_frames.append(img)
                depth_frames.append(depth_img)
                heatmap_frames.append(heatmap)

        # From here on, just stuff used for display
        img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img_norm = np.uint8(img_norm * 255)
        
        depth_norm = cv2.normalize(depth_img, None, 0, 1, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm * 255)

        if not started:
            draw_debug(heatmap, initial_pose, None, tracker, 1, debug_info)

        heatmap = cv2.resize(heatmap, (0,0), fx=2.2, fy=2.2)
        cv2.imshow('depth', depth_norm)
        cv2.imshow('heatmap', heatmap)
        cv2.imshow('color', img_norm)

        if DEBUG_TIME:
            print('time: ', time.time() - last_time)
            last_time = time.time()

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

    sensor.close()
    cv2.destroyAllWindows()

    if save_frames:
        for i in tqdm(range(len(heatmap_frames))):
            #cv2.imwrite(os.path.join(output_path, f'{i}.png'), heatmap_frames[i])
            cv2.imwrite(os.path.join(output_path, f'{i}d.png'), depth_frames[i])

    if save_video:
        for i in range(len(heatmap_frames)):
            out.write(heatmap_frames[i])
        out.release()