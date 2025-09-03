import os
import time

import cv2
import numpy as np
import keyboard
from tqdm import tqdm
import types
import importlib.machinery
from pynput.mouse import Listener
import torch


from pykinect2.PyKinectV2 import _DepthSpacePoint
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.utils import compute_axis, image_blend
from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch
from deep_6dof_tracking.image_show import ImageShow, ImageShowMessage, ESCAPE_KEY
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.utils import show_frames, normalize_scale, combine_view_transform, compute_2Dboundingbox



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


# https://github.com/KonstantinosAng/PyKinect2-Mapper-Functions/blob/master/mapper.py (modified)
# def depth_2_color_space(kinect, depth_space_point, depth_frame_data, show=False, return_aligned_image=False):
def get_frame_in_color_space(kinect, depth_space_point, depth_frame_data, show=False, return_aligned_image=False):
    """
    :param kinect: kinect class
    :param depth_space_point: _DepthSpacePoint from PyKinectV2
    :param depth_frame_data: kinect._depth_frame_data
    :param show: shows the aligned image
    :return: return the mapped color frame to depth frame
    """
    # Import here to optimize
    import numpy as np
    import ctypes
    import cv2
    # Map Color to Depth Space
    color2depth_points_type = depth_space_point * int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))

    depth_frame_data = depth_frame_data.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)
    # Where color_point = [xcolor, ycolor]
    # color_x = color2depth_points[depth_point[1] * 1920 + color_point[0] - 1].x
    # color_y = color2depth_points[depth_point[1] * 1920 + color_point[0] - 1].y
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(kinect.color_frame_desc.Height*kinect.color_frame_desc.Width,)))  # Convert ctype pointer to array
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 2).astype(int)
    depthXs = np.clip(depthXYs[:, :, 0], 0, kinect.depth_frame_desc.Width - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, kinect.depth_frame_desc.Height - 1)
    if (show or return_aligned_image):
        depth_frame = kinect.get_last_depth_frame()
        color_frame = kinect.get_last_color_frame()
        color_img = color_frame.reshape(((color_height, color_width, 4))).astype(np.uint8)
        depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width, 1)).astype(np.uint16)
        align_depth_img = np.zeros((1080, 1920), dtype=np.uint16)
        align_depth_img[:, :] = depth_img[depthYs, depthXs, 0]
        if show:
            cv2.imshow('Aligned Image', cv2.resize(cv2.flip(align_depth_img, 1), (int(1920 / 2.0), int(1080 / 2.0))))
            cv2.waitKey(3000)
        if return_aligned_image:
            return color_img, align_depth_img
    return depthXs, depthYs

# started = False
# def on_click(x, y, button, pressed):
#     global started
#     if pressed:
#         print(f"Mouse clicked at ({x}, {y}) with {button}")
#         started = True
#     return False

def on_mouse_click(event, x, y, flags, param):
    global started
    if event == cv2.EVENT_LBUTTONDOWN:
        started = True

# Set up variables
started = False

if __name__ == '__main__':

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
    new_renderer = configs["new_renderer"]
    bb3d = configs["bb3d"]

    camera = Camera.load_from_json(camera_path)
    # https://github.com/limgm/PyKinect2/tree/master
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                            PyKinectV2.FrameSourceTypes_Depth)
    depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height # Default: 512, 424
    color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height # Default: 1920, 1080

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, 'video.mp4'), fourcc, 11.0, (1920, 1080))
    color_frames = []
    depth_frames = []

    initial_pose = Transform.from_parameters(0, 0, -1, -2.355, 0, 0)
    reset_pose = True
    last_time = time.time()

    cv2.namedWindow('color')
    cv2.namedWindow('depth')
    cv2.setMouseCallback('color', on_mouse_click)

    from deep_6dof_tracking.networks.deeptrack_res_net import DeepTrackResNet
    tracker = DeepTrackerBatch(camera, backend, DeepTrackResNet, new_renderer=new_renderer, bb3d=bb3d) # Carefull with the network and bb3d
    tracker.load(os.path.join(network_path, "model_best.pth.tar"), None, geometry_path, None, shader_path)



    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    


    continue_loop = True
    previous_pose = initial_pose
    print('waiting for click')
    #with Listener(on_click=on_click) as listener:
    while continue_loop:
        if kinect.has_new_color_frame() and \
        kinect.has_new_depth_frame():

            depth_frame = kinect.get_last_depth_frame()
            depth_img = depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16)
            color_img, depth_img = get_frame_in_color_space(kinect, _DepthSpacePoint , depth_img, return_aligned_image=True, show=False)

            if started:
                # if save_frames:
                color_frames.append(color_img)
                depth_frames.append(depth_img)
                
                #if save_video:
                    #color_img_reshape = color_img.reshape((1080, 1920, 4))
                    #out.write(color_img_reshape[:,:,:3])
                reset_pose = False

            interval = time.time() - last_time
            print(interval)
            last_time = time.time()


            
            bb = compute_2Dboundingbox(previous_pose, camera, tracker.object_width, scale=(1000, 1000, -1000))
            rgbA, depthA = tracker.compute_render(previous_pose, bb)
            debug_info = (rgbA, bb, np.hstack((rgbA, rgbA)))


            estimation_time = time.time()
            if reset_pose:
                previous_pose = initial_pose
            else:
                for j in range(iterations):
                    time_beg = time.time()
                    predicted_pose, debug_info = tracker.estimate_current_pose(previous_pose, color_img[:,:,:3], depth_img,
                                                                            verbose=False,
                                                                            debug_time=False,
                                                                            batch=False,
                                                                            iteration=j)
                    previous_pose = predicted_pose
            estimation_time = time.time() - estimation_time

            print(f'estimation_time: {estimation_time}')
            draw_debug(color_img[:,:,:3], previous_pose, None, tracker, 1, debug_info)

            color_img_resize = cv2.resize(color_img, (0,0), fx=0.5, fy=0.5) # Resize (1080, 1920, 4) into half (540, 960, 4)
            depth_img_resize = cv2.resize(depth_img, (0,0), fx=0.5, fy=0.5) # Resize (1080, 1920, 4) into half (540, 960, 4)
            depth_img = cv2.convertScaleAbs(depth_img_resize, alpha=255/1500)
            depth_img = depth_img.astype(np.uint8)
            depth_colormap   = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET) # Scale to display from 0 mm to 1500 mm
            
            
            cv2.imshow('color', color_img_resize)                       # (540, 960, 4)
            cv2.imshow('depth', depth_colormap)                         # (424, 512)

        key = cv2.waitKey(1)
        if keyboard.is_pressed('esc'):
            continue_loop = False
        if key == ESCAPE_KEY:
            continue_loop = False


    print('writing frames')

    if save_frames:
        for i in tqdm(range(len(color_frames))):
            cv2.imwrite(os.path.join(output_path, f'{i}.png'), color_frames[i])
            cv2.imwrite(os.path.join(output_path, f'{i}d.png'), depth_frames[i])

    if save_video:
        for i in range(len(color_frames)):

            color_frame = color_frames[i].reshape((1080, 1920, 4))
            out.write(color_frame[:,:,:3])
        print('saving video...')
        out.release()

    kinect.close()
    cv2.destroyAllWindows()