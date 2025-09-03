from pytorch_toolbox.io import yaml_load
from tqdm import tqdm

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.sensors.viewpointgenerator import ViewpointGenerator
from deep_6dof_tracking.utils.aruco import ArucoDetector
from deep_6dof_tracking.data.sensors.kinect2 import Kinect2
from deep_6dof_tracking.data.utils import compute_2Dboundingbox, image_blend, normalize_scale
from deep_6dof_tracking.utils.icp import icp
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.frame import Frame
from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
import sys
import json
import os
import cv2
import math
import numpy as np
import time

ESCAPE_KEY = 1048603
NUM_PAD_1_KEY = 1114033
NUM_PAD_2_KEY = 1114034
NUM_PAD_3_KEY = 1114035
NUM_PAD_4_KEY = 1114036
NUM_PAD_5_KEY = 1114037
NUM_PAD_6_KEY = 1114038
NUM_PAD_7_KEY = 1114039
NUM_PAD_8_KEY = 1114040
NUM_PAD_9_KEY = 1114041
NUM_PLUS_KEY = 1114027
NUM_MINUS_KEY = 1114029
ARROW_LEFT_KEY = 1113937
ARROW_UP_KEY = 1113938
ARROW_RIGHT_KEY = 1113939
ARROW_DOWN_KEY = 1113940


def lerp(value, maximum, start_point, end_point):
    return start_point + (end_point - start_point) * value / maximum


def show_occlusion(detection, rgb, depth, camera, bb_width):
    pixels = compute_2Dboundingbox(detection, camera, bb_width)
    depth_crop = depth[pixels[0, 0]:pixels[1, 0], pixels[0, 1]:pixels[2, 1]].astype(float)
    mask = np.bitwise_and(depth_crop < 880, depth_crop != 0)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3)))
    print("Occlusion level : {}".format(np.sum(mask) / (mask.shape[0] * mask.shape[1])))
    cv2.imshow("object crop mask", (mask * 255))
    cv2.imshow("object crop depth", ((depth_crop / np.max(depth_crop) * 255).astype(np.uint8)))
    cv2.rectangle(rgb, tuple(pixels[0][::-1]), tuple(pixels[3][::-1]), (0, 0, 255), 2)


def clean_point_cloud(points):
    # remove zeros
    points = points[np.all(points != 0, axis=1)]

    return points


def crop_point_cloud(points, radius=0.15):
    # board data only
    points = points[points[:, 0] < radius]
    points = points[points[:, 0] > -radius]
    points = points[points[:, 1] < radius]
    points = points[points[:, 1] > -radius]
    points = points[points[:, 2] > 0.01]
    return points


def transform_pointcloud(points, pose):
    transform = pose.inverse()
    scale = Transform.scale(1, -1, -1)
    transform.combine(scale)
    points = transform.rotation.dot(points)
    points = transform.translation.dot(points)
    return points

def register_pointclouds(cloud1, cloud2):
    frame_points = clean_point_cloud(cloud1)
    frame_points = transform_pointcloud(frame_points, detection)
    frame_points = crop_point_cloud(frame_points)

    render_points = clean_point_cloud(cloud2)
    PlyParser.save_points(render_points, "render.ply")
    render_points = transform_pointcloud(render_points, detection)

    diff_transform, _ = icp(frame_points, render_points, max_iterations=10, tolerance=0.1)
    return diff_transform

alpha = 1
def trackbar(x):
    global alpha
    alpha = x/100

if __name__ == '__main__':

    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "capture_config.yml"
    configs = yaml_load(config_path)

    MODEL_GEO_PATH = configs["model_geo_path"]
    MODEL_AO_PATH = configs["model_ao_path"]
    MODEL_WIDTH = configs["model_width"]
    SHADER_PATH = configs["shader_path"]
    OUTPUT_PATH = configs["output_path"]
    CAMERA_PATH = configs["camera_path"]
    DETECTOR_PATH = configs["detector_path"]
    PRELOAD = configs["preload"]
    SHOW_OCCLUSION = configs["show_occlusion"]
    SHOW_DEPTH = configs["show_depth"]
    SHOW_BB = configs["show_bb"]
    BOARD_TO_OBJECT_PATH = configs["board_to_object_path"]
    RECOMPUTE_POSE = configs["recompute_pose"]
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    sensor = Kinect2(CAMERA_PATH)
    detector = ArucoDetector(sensor.camera, DETECTOR_PATH)
    frame_generator = ViewpointGenerator(sensor, detector)

    dataset = DeepTrackLoaderBase(OUTPUT_PATH, read_data=PRELOAD)
    dataset.set_save_type("png")
    dataset.camera = sensor.camera
    window = InitOpenGL(sensor.camera.width, sensor.camera.height)
    vpRender = ModelRenderer(MODEL_GEO_PATH, SHADER_PATH, sensor.camera, window, (sensor.camera.width, sensor.camera.height))
    vpRender.load_ambiant_occlusion_map(MODEL_AO_PATH)

    cv2.namedWindow('image')
    cv2.createTrackbar('transparency', 'image', 0, 100, trackbar)

    # todo, read from file?
    detection_offset = Transform()
    rgbd_record = False
    save_next_rgbd_pose = False
    lock_offset = False
    if BOARD_TO_OBJECT_PATH:
        offset_path = os.path.join(BOARD_TO_OBJECT_PATH, "board2object.npy")
        detection_offset = Transform.from_matrix(np.load(offset_path))
        lock_offset = True

    for i, (current_frame, ground_truth_pose) in enumerate(frame_generator):
        start_time = time.time()
        bgr, depth = current_frame.get_rgb_depth(None)
        screen = bgr.copy()

        if rgbd_record:
            # here we add a dummy pose, we will compute the pose as a post operation
            dataset.add_pose(bgr, depth, Transform())
        else:
            detection = detector.detect(screen)
            # Draw a color rectangle around screen : red no detection, green strong detection
            color_ = lerp(detector.get_likelihood(), 1, np.array([255, 0, 0]), np.array([0, 255, 0]))
            cv2.rectangle(screen, (0, 0), (sensor.camera.width, sensor.camera.height), tuple(color_), 10)
            if detection:
                # Add objects offset
                detection.combine(detection_offset.inverse())

                if SHOW_BB:
                    bb = compute_2Dboundingbox(detection, sensor.camera, MODEL_WIDTH,
                                                     scale=(1000, -1000, -1000))
                    screen_bb, depthB = normalize_scale(screen, depth, bb, (150, 150))
                    cv2.imshow("bb", screen_bb[:, :, ::-1])

                if SHOW_OCCLUSION:
                    show_occlusion(detection, screen, depth, sensor.camera, MODEL_WIDTH)
                if save_next_rgbd_pose:
                    dataset.add_pose(bgr, depth, detection)
                    save_next_rgbd_pose = False
                rgb_render, depth_render = vpRender.render_image(detection)
                bgr_render = rgb_render[:, :, ::-1].copy()
                blend = image_blend(bgr_render, screen)
                screen = cv2.addWeighted(screen, 1 - alpha, blend, alpha, 1)

        cv2.putText(screen, "Fps : {:10.4f}".format(1./(time.time() - start_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("image", screen[:, :, ::-1])
        if SHOW_DEPTH:
            cv2.imshow("depth", (depth[:, :] / np.max(depth) * 255).astype(np.uint8))
        key = cv2.waitKey(1)
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key == ESCAPE_KEY:
            break
        elif key_chr == 'r':
            rgbd_record = not rgbd_record
        elif key_chr == ' ':
            save_next_rgbd_pose = True
        # Lock offset makes sure that we wont change the file from an already generated dataset... It is important
        # since we do not want to have a different offset for each pictures. offset file is only used to compute
        # ground truth object pose given images
        if not lock_offset:
            if key == NUM_PAD_1_KEY:
                detection_offset.rotate(z=math.radians(-1))
            elif key == NUM_PAD_2_KEY:
                detection_offset.translate(z=0.001)
            elif key == NUM_PAD_3_KEY:
                detection_offset.rotate(x=math.radians(-1))
            elif key == NUM_PAD_4_KEY:
                detection_offset.translate(x=-0.001)
            elif key == NUM_PAD_5_KEY:
                frame_points = sensor.camera.backproject_depth(depth)/1000
                render_points = sensor.camera.backproject_depth(depth_render)/1000
                new_offset = register_pointclouds(frame_points, render_points)
                detection_offset.combine(new_offset)

            elif key == NUM_PAD_6_KEY:
                detection_offset.translate(x=0.001)
            elif key == NUM_PAD_7_KEY:
                detection_offset.rotate(z=math.radians(1))
            elif key == NUM_PAD_8_KEY:
                detection_offset.translate(z=-0.001)
            elif key == NUM_PAD_9_KEY:
                detection_offset.rotate(x=math.radians(1))
            elif key == ARROW_UP_KEY:
                detection_offset.translate(y=-0.001)
            elif key == ARROW_DOWN_KEY:
                detection_offset.translate(y=0.001)
            elif key == ARROW_LEFT_KEY:
                detection_offset.rotate(y=math.radians(-1))
            elif key == ARROW_RIGHT_KEY:
                detection_offset.rotate(y=math.radians(1))

        if key_chr == "s":
            del frame_generator
            print("Compute detections")
            last_valid_pose = Transform()
            for i in range(dataset.size()):
                frame, pose = dataset.data_pose[i]
                # if pose is identity, compute the detection
                if pose == Transform() or RECOMPUTE_POSE:
                    rgb, depth = dataset.data_pose[i][0].get_rgb_depth(dataset.root)
                    pose = detector.detect(rgb)
                    if pose is None:
                        # Handle bad aruco readings
                        pose = last_valid_pose
                    else:
                        last_valid_pose = pose
                    pose.combine(detection_offset.inverse())
                    if detector.get_likelihood() < 0.1:
                        print("[WARNING] : Detector returns uncertain pose at frame {}".format(i))
                    #Todo : need better way to handle viewpoint's pose change in dataset...
                    if RECOMPUTE_POSE:
                        rgb = None
                        depth = None
                    dataset.data_pose[i] = (Frame(rgb, depth, str(i)), pose)
            np.save(os.path.join(dataset.root, "board2object"), detection_offset.matrix)
            dataset.dump_images_on_disk()
            dataset.save_json_files({"save_type": "png"})
            break
