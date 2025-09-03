import argparse
import configparser

from deep_6dof_tracking.data.sequence_loader import SequenceLoader, Sequence
from deep_6dof_tracking.eccv.eval_functions import get_pose_difference, compute_pose_diff
import time
import cv2
import numpy as np
import os

from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.data_logger import DataLogger
from deep_6dof_tracking.utils.draw import draw_axis, draw_color_blend
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import compute_axis, image_blend

import pandas as pd

ESCAPE_KEY = 27
DEPTHONLY = True

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')

    parser.add_argument('-o', '--output', help="Output path", metavar="FILE")
    parser.add_argument('-s', '--sequence', help="Sequence path", metavar="FILE")
    parser.add_argument('-m', '--model', help="Network model path", metavar="FILE")
    parser.add_argument('-g', '--geometry', help="3D model path", metavar="FILE")
    parser.add_argument('--results', help="Results path", metavar="FILE")
    parser.add_argument('--shader', help="shader path", action="store", default="data/shaders")
    parser.add_argument('--cascade', help="cascade refinement model", action="store", default="None")

    parser.add_argument('-v', '--verbose', help="Print information during runtime", action="store_true")
    parser.add_argument('-d', '--debug', help="Show debug information", action="store_true")
    parser.add_argument('--show', help="show screen", action="store_true")
    parser.add_argument('-sv', '--video', help="Encode video during runtime", action="store_true")
    parser.add_argument('-sf', '--frame', help="Save frames during runtime", action="store_true")
    parser.add_argument('-x', '--axis', help="Show axis instead of 3D model", action="store_true")
    parser.add_argument('-t', '--time', help="Print time information", action="store_true")
    parser.add_argument('-b', '--batch', help="Enable batch mode", action="store_true")
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('--rgb', help="remove depth data", action="store_true")

    parser.add_argument('-a', '--architecture', help="architecture name", action="store", default="squeeze_large")
    parser.add_argument('-i', '--iteration', help="iteration per prediction", action="store", default=3, type=int)
    parser.add_argument('--object_width', help="iteration per prediction", action="store", default=0, type=int)
    parser.add_argument('-r', '--reset', help="Number of frames before reseting tracker",
                        action="store", default=0, type=int)
    parser.add_argument('--resetlost', help="Reset the tracker when it fails", action="store_true")
    parser.add_argument('--nogt', help="No ground truth poses", action="store_true")
    parser.add_argument('--depthonly', help="depth only", action="store_true")
    parser.add_argument('--bb3d', help="crop the observed data with a 3D bounding box", action="store_true")
    parser.add_argument('--same_mean', help="use the same mean and std for render and obs data", action="store_true")
    parser.add_argument('--new_renderer', help='Use new renderer for obj files', action="store_true")

    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--pose', type=str, help='starting pose')

    arguments = parser.parse_args()

    if arguments.config:
        config = configparser.ConfigParser()
        config.read(arguments.config)

        # Update arguments from the config file
        for key in config['Arguments']:
            value = config['Arguments'][key]
            
            # Handle type conversion for specific arguments
            if key in ['iteration', 'object_width', 'reset']:
                setattr(arguments, key, int(value))
            elif key in ['verbose', 
                         'debug', 
                         'show', 
                         'video', 
                         'frame', 
                         'axis', 
                         'time',
                         'batch', 
                         'rgb', 
                         'resetlost', 
                         'nogt', 
                         'depthonly',
                         'bb3d',
                         'same_mean',
                         'new_renderer']:
                setattr(arguments, key, config.getboolean('Arguments', key))
            else:
                setattr(arguments, key, value)

    return arguments


def draw_debug(img, pose, tracker, alpha, debug_info, outline=False):
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

        axis = compute_axis(pose, tracker.camera)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, 255), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, 255, 0), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (255, 0, 0), 3)
        
        # Crop the image
        img_render = img_render[y_min:, x_min:, :]
        if outline:
            obj_mask = np.zeros_like(img_render, dtype=np.uint8)
            obj_mask[img_render > 0] = 255
            obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_RGB2GRAY)
            obj_mask_new = cv2.dilate(obj_mask, np.ones((5, 5), np.uint8), iterations=1)
            obj_mask_new[obj_mask > 0] = 0
            obj_mask_new = cv2.dilate(obj_mask_new, np.ones((5, 5), np.uint8), iterations=1)
            obj_mask_color = np.zeros((img_render.shape[0], img_render.shape[1], 3), dtype=np.uint8)
            obj_mask_color[obj_mask_new > 0] = [0, 255, 0]
            blend = image_blend(obj_mask_color[:h, :w, ::-1], crop)
        else:
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
            
    else:
        axis = compute_axis(pose, tracker.camera)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, 255), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, 255, 0), 3)
        cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (255, 0, 0), 3)



def execution_loop(current_rgb, current_depth, previous_pose, data_logger):
    debug_info = None
    # process pose estimation of current frame given last pose
    start_time = time.time()
    for j in range(int(CLOSED_LOOP_ITERATION)):
        batch = False
        if j != 0 and j == CLOSED_LOOP_ITERATION-1:
            batch = True
        if not BATCH_MODE:
            batch = False
        current_rgb = np.zeros((current_depth.shape[0], current_depth.shape[1], 3)) # mettre rgb a 0
        predicted_pose, debug_info = tracker.estimate_current_pose(previous_pose, current_rgb, current_depth,
                                                                    verbose=VERBOSE, batch=batch, debug_time=DEBUG_TIME,
                                                                    debug_show=DEBUG, iteration=j, rgb_only=RGB_ONLY)
        previous_pose = predicted_pose
    if DEBUG_TIME:
        print("Estimation processing time : {}".format(time.time() - start_time))

    return previous_pose, debug_info

def gen(alist):
    for i in alist:
        yield i

def dataset_loop(video_path):
    # frame_folder = os.path.join(OUTPUT_PATH, "videos", video_path.split("/")[-1])
    frame_folder = OUTPUT_PATH
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(frame_folder, "video.mp4"), fourcc, 15.0, (camera.width, camera.height))
        # out = cv2.VideoWriter(os.path.join(frame_folder, "video.avi"), fourcc, 27.0, (camera.width, camera.height))

    print("Processing : {}".format(video_path))
    video_data = Sequence(video_path, depthonly=DEPTHONLY, preload=True)

    if STARTING_POSE == "cam1":
        previous_pose = Transform.from_parameters(-0.05, 0.425, -1.2, -1.2, 0.3, 0.4) # CAM1
    elif STARTING_POSE == "cam2":
        previous_pose = Transform.from_parameters(-0.75, 0.3, -1.75, -1.7, -0.6, -0.25) # CAM2
    

    pose_pred_logger = DataLogger()
    pose_pred_logger.create_dataframe("prediction_pose", ("M1", "M2", "M3", "M4",
                                                          "M5", "M7", "M8", "M9",
                                                          "M10", "M11", "M12", "M13",
                                                          "M14", "M15", "M16", "M17"))
    data_logger = DataLogger()
    data_logger.create_dataframe("projection_error", ("error",))
    print("Run {}".format(video_path))
    reset_counter = 0
    total_resets = 0
    df = pd.DataFrame(columns=['translation_error(meters)', 'rotation_error(degrees)'])

    # start_pose = previous_pose.copy()

    for i in range(video_data.size()):
        # get actual frame
        # current_rgb, current_depth = current_frame.get_rgb_depth(video_data.root)
        #current_depth[:, :] = 0
        current_rgb, current_depth = video_data.load_image(i)
        screen = current_rgb.copy() if not DEPTHONLY else current_depth.copy()
        
        if DEPTHONLY:
            screen = cv2.convertScaleAbs(screen, alpha=0.05)
            screen = cv2.applyColorMap(np.uint8(screen), cv2.COLORMAP_INFERNO)
        
        heatmap = cv2.convertScaleAbs(current_depth, alpha=0.05)
        heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_INFERNO)

        # set_gt_pose = RESET_FREQUENCY != 0 and i % RESET_FREQUENCY == 0
        # if RESET_LOST and reset_counter > 7:
        #     reset_counter = 0
        #     total_resets += 1
        #     set_gt_pose = True

        current_rgb = np.zeros_like(current_depth) if DEPTHONLY else current_rgb
        previous_pose, debug_info = execution_loop(current_rgb, current_depth, previous_pose,
                                                   data_logger)

        # detect tracking fail, and triger reset
        # diff = get_pose_difference(previous_pose, ground_truth_pose)
        # pose_diff = diff[np.newaxis, :]
        # diff_t, diff_r = compute_pose_diff(pose_diff)

        # df = df.append({'translation_error(meters)': diff_t,
        #                 'rotation_error(degrees)': diff_r},
        #                ignore_index=True)
        # df.loc[i] = [diff_t, diff_r]

        # error_detected = False
        # error_diff_t = np.sqrt(np.square(pose_diff[:, 0]) + np.square(pose_diff[:, 1]) + np.square(pose_diff[:, 2]))
        # if error_diff_t > 0.03:
        #     error_detected = True
        # if diff_r > 20:
        #     error_detected = True
        # reset_counter = reset_counter + 1 if error_detected else 0

        if SHOW_AXIS:
            debug_info = None
        # pose_gt_logger.add_row(pose_gt_logger.get_dataframes_id()[0], ground_truth_pose.matrix.flatten())
        pose_pred_logger.add_row(pose_pred_logger.get_dataframes_id()[0], previous_pose.matrix.flatten())
        

        # previous_pose = start_pose.copy()

        draw_debug(screen, previous_pose, tracker, 0.8, debug_info, outline=True)
        if SAVE_VIDEO:
            saved = screen[:, :, ::-1] if not DEPTHONLY else screen
            out.write(saved)
            # out.write(heatmap)
        if SAVE_FRAMES:
            # if not DEPTHONLY:
            #     cv2.imwrite(os.path.join(frame_folder, "rgb_{}.jpg".format(i)), current_rgb[:, :, ::-1])
            # cv2.imwrite(os.path.join(frame_folder, "depth_{}.jpg".format(i)), heatmap)
            cv2.imwrite(os.path.join(frame_folder, "{}.jpg".format(i)), screen[:, :, ::-1])
        if SHOW:
            
            #cv2.imshow("Debug", cv2.resize(screen[:, :, ::-1], (0, 0), fx=0.5, fy=0.5))
            cv2.imshow("Debug", cv2.resize(screen[:, :, ::-1], (0, 0), fx=2, fy=2))
            key = cv2.waitKey(1)
            key_chr = chr(key & 255)
            if key != -1:
                print("pressed key id : {}, char : [{}]".format(key, key_chr))
            if key_chr == " ":
                print("Reset at frame : {}".format(i))
                # previous_pose = ground_truth_pose
                detection_mode = not detection_mode
            if key == ESCAPE_KEY:
                break

    # mean_translation_error = df['translation_error(meters)'].mean()
    # mean_rotation_error = df['rotation_error(degrees)'].mean()
    # print(f"Mean translation error: {mean_translation_error} meters")
    # print(f"Mean rotation error: {mean_rotation_error} degrees")
    # lines_to_append = [
    #     f"translation error: {mean_translation_error[0]}\n",
    #     f"rotation error: {mean_rotation_error[0]}\n",
    #     f"failures: {total_resets}\n\n"
    # ]

    # with open(RESULTS_PATH, 'a') as file:
    #     for line in lines_to_append:
    #         file.write(line)

    data_logger.save(OUTPUT_PATH)
    # pose_gt_logger.save(OUTPUT_PATH)
    pose_pred_logger.save(OUTPUT_PATH)
    if SAVE_VIDEO:
        out.release()
    print(f'Total resets : {total_resets}')

if __name__ == '__main__':
    arguments = parse_args()
    
    VERBOSE = arguments.verbose
    SAVE_VIDEO = arguments.video
    SAVE_FRAMES = arguments.frame
    SHOW_AXIS = arguments.axis
    DEBUG_TIME = arguments.time
    ARCHITECTURE = arguments.architecture
    BATCH_MODE = arguments.batch
    BACKEND = arguments.backend
    CLOSED_LOOP_ITERATION = arguments.iteration
    RESET_FREQUENCY = arguments.reset
    RESET_LOST = arguments.resetlost
    SHOW = arguments.show
    DEBUG = arguments.debug
    RGB_ONLY = arguments.rgb
    OBJECT_WIDTH = arguments.object_width
    if OBJECT_WIDTH == 0:
        OBJECT_WIDTH = None
    BB3D = arguments.bb3d
    SAME_MEAN = arguments.same_mean
    NEW_RENDERER = arguments.new_renderer

    OUTPUT_PATH = arguments.output
    VIDEO_PATH = arguments.sequence
    MODEL_PATH = arguments.model
    GEOMETRY_PATH = arguments.geometry
    SHADER_PATH = arguments.shader
    RESULTS_PATH = arguments.results
    CASCADE_PATH = arguments.cascade
    if CASCADE_PATH == "None":
        CASCADE_PATH = None
    NOGT = arguments.nogt
    depthonly = arguments.depthonly
    STARTING_POSE = arguments.pose

    if 'interaction_hard' in VIDEO_PATH and RESET_FREQUENCY != 0:
        RESET_FREQUENCY = 0
        RESET_LOST = True

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    video_folder = os.path.join(OUTPUT_PATH, "videos")
    # if not os.path.exists(video_folder):
    #     os.mkdir(video_folder)

    model_split_path = MODEL_PATH.split(os.sep)
    model_folder = os.sep.join(model_split_path[:-1])

    MODEL_3D_PATH_GEO = os.path.join(GEOMETRY_PATH, "geometry.ply")
    MODEL_3D_PATH_AO = os.path.join(GEOMETRY_PATH, "ao.ply")
    if not os.path.exists(MODEL_3D_PATH_AO):
        MODEL_3D_PATH_AO = None

    camera = Camera.load_from_json(VIDEO_PATH)
    # Makes the list a generator for compatibility with sensor's generator

    tracker = DeepTrackerBatch(camera,
                               BACKEND,
                               ARCHITECTURE,
                               bb3d=BB3D,
                               same_mean=SAME_MEAN,
                               new_renderer=NEW_RENDERER)
    tracker.load(MODEL_PATH, MODEL_3D_PATH_GEO.split("/")[-2], MODEL_3D_PATH_GEO, MODEL_3D_PATH_AO, SHADER_PATH, OBJECT_WIDTH)
    model_3d = PlyParser(MODEL_3D_PATH_GEO).get_vertex()
    dataset_loop(VIDEO_PATH)

