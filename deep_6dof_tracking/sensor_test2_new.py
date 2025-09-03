import time
import os

import cv2
import numpy as np
from tqdm import tqdm
import imageio

from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.data.utils import compute_axis, image_blend, compute_2Dboundingbox
from deep_6dof_tracking.data.sensor_wrapper import SensorWrapper, SensorType
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch

import matplotlib.pyplot as plt

def draw_debug(img, pose, gt_pose, tracker, alpha, debug_info):
    if debug_info is not None:
        img_render, bb, _ = debug_info
        img_render = cv2.resize(img_render, (bb[2, 1] - bb[0, 1], bb[1, 0] - bb[0, 0]))
        bb_copy = bb.copy()
        bb[bb<0] = 0
        crop = img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :]

        h, w, c = crop.shape
        x_min = bb_copy[0, 1]*-1 if bb_copy[0, 1] < 0 else 0
        y_min = bb_copy[0, 0]*-1 if bb_copy[0, 0] < 0 else 0
        x_max = min(w, bb_copy[2, 1])
        y_max = min(h, bb_copy[1, 0])
        
        # Crop the image
        img_render = img_render[y_min:, x_min:, :]
        blend = image_blend(img_render[:h, :w, ::-1], crop)
        axis = compute_axis(pose, tracker.camera)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, 255), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, 255, 0), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (255, 0, 0), 3)
        try:
            bb[bb<0] = 0
            img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :] = cv2.addWeighted(img[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[2, 1], :],
                                                                       1 - alpha, blend, alpha, 1)
        except Exception as e:
            print(e)
            axis = compute_axis(pose, tracker.camera)
            cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, 255), 3)
            cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, 255, 0), 3)
            cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (255, 0, 0), 3)
            if gt_pose is not None:
                axis_gt = compute_axis(gt_pose, tracker.camera)
                cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[1, ::-1]), (0, 0, 155), 3)
                cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[2, ::-1]), (0, 155, 0), 3)
                cv2.line(img, tuple(axis_gt[0, ::-1]), tuple(axis_gt[3, ::-1]), (155, 0, 0), 3)
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
    bb3d = configs["bb3d"]
    same_mean = configs["same_mean"]
    new_renderer = configs["new_renderer"]

    #Getting camera model
    camera = Camera.load_from_json(camera_path)

    # Initializing tracker
    from deep_6dof_tracking.networks.deeptrack_res_net import DeepTrackResNet
    # Architecture is always ResNet WARNING
    tracker = DeepTrackerBatch(camera, backend, DeepTrackResNet, bb3d=bb3d, same_mean=same_mean, new_renderer=new_renderer)
    tracker.load(os.path.join(network_path, "model_best.pth.tar"), None, geometry_path, None, shader_path)

    # Initializing camera
    sensor = SensorWrapper(SensorType.HELIOS2)
    # sensor.initialize(serial='242100863')
    sensor.initialize(serial='242100857')
    sensor.start_stream()

    # Creating windows for diplay
    # cv2.namedWindow('color')
    cv2.namedWindow('depth')
    cv2.namedWindow('heatmap')
    cv2.setMouseCallback('heatmap', on_mouse_click)

    initial_pose = Transform.from_parameters(-0.55, 0.5, -2.4, -1.7, -0.7, -0.5)


    # If we want to save video/frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, 'video.mp4'), fourcc, 11.0, (640, 480))
    color_frames = []
    depth_frames = []
    heatmap_frames = []

    # debug time is harcoded
    last_time = time.time()
    DEBUG_TIME=False
    reset_pose = True
    n_frames = 0
    total_time = 0
    total_capture_time = 0
    total_prediction_time = 0
    total_display_time = 0
    while True:
        capture_time = time.time()
        _, depth_img, heatmap = sensor.get_last_frame(heatmap=True)
        capture_time = time.time() - capture_time
        if DEBUG_TIME:
            print('capture_time:  ', capture_time)

        if started:
            reset_pose = False
            # In my case, I want to save a video of the heatmap
            # color_frames.append(img)
            # depth_frames.append(depth_img)
            heatmap_frames.append(heatmap)
            depth_frames.append(depth_img)


        # Compute a render of the object (will be used by the network)
        if not started:
            bb = compute_2Dboundingbox(initial_pose, camera, tracker.object_width, scale=(1000, 1000, -1000))
            rgbA, depthA = tracker.compute_render(initial_pose, bb)
            bb2 = compute_2Dboundingbox(initial_pose, camera, tracker.object_width, scale=(1000, -1000, -1000))
            debug_info = (rgbA, bb2, np.hstack((rgbA, rgbA)))
            

        # Pose prediction
        prediction_time = time.time()
        if reset_pose:
            previous_pose = initial_pose
        else:
            for j in range(iterations):
                time_beg = time.time()
                try:
                    predicted_pose, debug_info = tracker.estimate_current_pose(previous_pose, np.zeros_like(heatmap), depth_img,
                                                                            verbose=False,
                                                                            debug_time=False,
                                                                            batch=False,
                                                                            iteration=j,
                                                                            debug_show=False)
                except:
                    break
                
                predicted_pose.matrix # This is the transformation matrix between the camera and the center of the object
                previous_pose = predicted_pose
        prediction_time = time.time() - prediction_time
        if DEBUG_TIME:
            print('prediction_time:  ', prediction_time)
        
        # From here on, just stuff used for display
        display_time = time.time()
        # img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        # img_norm = np.uint8(img_norm * 255)
        
        depth_norm = cv2.normalize(depth_img, None, 0, 1, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm * 255)

        draw_debug(heatmap, previous_pose, None, tracker, 1, debug_info)

        heatmap = cv2.resize(heatmap, (0,0), fx=2.2, fy=2.2)
        cv2.imshow('depth', depth_norm)
        cv2.imshow('heatmap', heatmap)
        # cv2.imshow('color', img_norm)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        loop_time = time.time() - last_time
        display_time = time.time() - display_time
        if DEBUG_TIME:
            print('display_time: ', display_time)
            print('total time: ', loop_time)
            print()
        if started:
            total_time += loop_time
            total_capture_time += capture_time
            total_prediction_time += prediction_time
            total_display_time += display_time
            n_frames += 1
        last_time = time.time()


    # if DEBUG_TIME:
    mean_time = total_time/n_frames
    print('mean capture time:  ', total_capture_time/n_frames)
    print('mean prediction time:  ', total_prediction_time/n_frames)
    print('mean display time:  ', total_display_time/n_frames)
    print('mean time:  ', mean_time)
    print('fps:  ', 1/mean_time)

    sensor.close()

    if save_frames:
        for i in tqdm(range(len(depth_frames))):
            #cv2.imwrite(os.path.join(output_path, f'{i}.png'), heatmap_frames[i])
            # print(depth_frames[i].dtype)
            # cv2.imwrite(os.path.join(output_path, f'{i}d.png'), depth_frames[i])
            imageio.imwrite(os.path.join(output_path, f'{i}d.png'), depth_frames[i].astype(np.uint16))

    if save_video:
        for i in range(len(heatmap_frames)):
            out.write(heatmap_frames[i])
        out.release()