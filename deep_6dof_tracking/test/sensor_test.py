from deep_6dof_tracking.data.sensors.kinect2 import Kinect2
from deep_6dof_tracking.data.sensors.viewpointgenerator import ViewpointGenerator

import cv2
import time
import numpy as np
import os

ESCAPE_KEY = 1048603
SPACE_KEY = 1048608

if __name__ == '__main__':

    # Populate important data from config file
    CAMERA_PATH = "/home/mathieu/source/deep_6dof_tracking/deep_6dof_tracking/data/sensors/camera_parameter_files/Kinect2_lab_small.json"

    sensor = Kinect2(CAMERA_PATH)
    frame_generator = ViewpointGenerator(sensor)
    camera = sensor.camera

    # Frames from the generator are in camera coordinate
    previous_frame, previous_pose, last_timestamp = next(frame_generator)
    previous_rgb, previous_depth = previous_frame.get_rgb_depth(None)
    time_end_process = time.time()
    for i, (current_frame, ground_truth_pose, current_timestamp) in enumerate(frame_generator):
        time_start_process = time.time()
        kinect_fps = current_timestamp - last_timestamp
        last_timestamp = current_timestamp
        grab_time = time_start_process - time_end_process
        fps = 1/grab_time
        print("Grab time : {}s / {} fps".format(grab_time, fps))
        print("Kinect fps : {}".format(1/kinect_fps))
        current_rgb, current_depth = current_frame.get_rgb_depth(None)
        cv2.imshow("image", current_rgb[:, :, ::-1])
        key = cv2.waitKey(1)
        key_chr = chr(key & 255)
        if key != -1:
            print("pressed key id : {}, char : [{}]".format(key, key_chr))
        if key == ESCAPE_KEY:
            break
        time_end_process = time.time()


