"""
Compare object depth with/without marker
"""
import os
import cv2
import numpy as np
import math
from PIL import Image
from deep_6dof_tracking.utils.transform import Transform
import matplotlib.pyplot as plt

from py_rgbd_grabber.kinect2 import Kinect2

from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.draw import draw_axis, draw_color_blend, blend_depth_error
from deep_6dof_tracking.utils.keyboard_def import *
from deep_6dof_tracking.utils.aruco import ArucoDetector
from deep_6dof_tracking.utils.camera import Camera


def capture(root_path):
    output_path = os.path.join(root_path, "output")
    model_3d = os.path.join(root_path, "model_3d/geometry.ply")
    shaders_path = "../data/shaders"
    aruco_file = os.path.join(root_path, "aruco.xml")
    camera_file = os.path.join(root_path, "camera.json")

    sensor = Kinect2()
    camera = Camera.load_from_json(camera_file)
    small_camera = camera.copy()
    small_camera.set_ratio(1.5)
    small_camera_size = (small_camera.width, small_camera.height)

    object_aruco_to_object_real = Transform()
    if os.path.exists(os.path.join(root_path, "calib.npy")):
        object_aruco_to_object_real = Transform.from_matrix(np.load(os.path.join(root_path, "calib.npy")))

    detector = ArucoDetector(small_camera, aruco_file)
    renderer = ModelRenderer2(model_3d, shaders_path, small_camera, [small_camera_size], backend="glfw")

    sensor.initialize_()
    count = 0
    while True:
        frame = sensor.get_frame_()
        #frame.depth -= 35
        rgb = cv2.resize(frame.rgb, small_camera_size)
        depth = cv2.resize(frame.depth, small_camera_size)


        pose = detector.detect(rgb)
        pose = pose.combine(object_aruco_to_object_real, copy=True)

        rgb_render, depth_render = renderer.render_image(pose)

        rgb = blend_depth_error(depth_render, depth, rgb)
        draw_axis(rgb, pose, small_camera)

        cv2.imshow("capture", rgb[:, :, ::-1])

        key = cv2.waitKey(1)
        if key > 0:
            print(key)
        if key == ESCAPE_KEY:
            break
        if key == SPACE_KEY:
            np.save(os.path.join(root_path, "calib.npy"), object_aruco_to_object_real.matrix)
            cv2.imwrite(os.path.join(output_path, '{}.png').format(count), frame.rgb[:, :, ::-1])
            cv2.imwrite(os.path.join(output_path, '{}d.png').format(count), frame.depth.astype(np.uint16))
            cv2.imwrite(os.path.join(output_path, '{}d_render.png').format(count), depth_render.astype(np.uint16))
            count += 1
        elif key == NUM_PAD_1_KEY:
            object_aruco_to_object_real.rotate(z=math.radians(-1))
        elif key == NUM_PAD_2_KEY:
            object_aruco_to_object_real.translate(z=0.001)
        elif key == NUM_PAD_3_KEY:
            object_aruco_to_object_real.rotate(x=math.radians(-1))
        elif key == NUM_PAD_4_KEY:
            object_aruco_to_object_real.translate(x=-0.001)
        elif key == NUM_PAD_6_KEY:
            object_aruco_to_object_real.translate(x=0.001)
        elif key == NUM_PAD_7_KEY:
            object_aruco_to_object_real.rotate(z=math.radians(1))
        elif key == NUM_PAD_8_KEY:
            object_aruco_to_object_real.translate(z=-0.001)
        elif key == NUM_PAD_9_KEY:
            object_aruco_to_object_real.rotate(x=math.radians(1))
        elif key == ARROW_UP_KEY:
            object_aruco_to_object_real.translate(y=-0.001)
        elif key == ARROW_DOWN_KEY:
            object_aruco_to_object_real.translate(y=0.001)
        elif key == ARROW_LEFT_KEY:
            object_aruco_to_object_real.rotate(y=math.radians(-1))
        elif key == ARROW_RIGHT_KEY:
            object_aruco_to_object_real.rotate(y=math.radians(1))
    sensor.clean_()


def repair(depth, render, marker_coords, pixel_window):
    for marker_coord in marker_coords:
        render_crop = render[marker_coord[1] - pixel_window:marker_coord[1] + pixel_window,
                             marker_coord[0] - pixel_window:marker_coord[0] + pixel_window].copy()
        depth_crop = depth[marker_coord[1] - pixel_window:marker_coord[1] + pixel_window,
                            marker_coord[0] - pixel_window:marker_coord[0] + pixel_window]
        l = 3
        render_blend = 0.75
        depth_blend = 1 - render_blend
        render_crop[:, 0:l] = render_blend * render_crop[:, 0:l] + depth_blend * depth_crop[:, 0:l]
        render_crop[:, -l:] = render_blend * render_crop[:, -l:] + depth_blend * depth_crop[:, -l:]
        render_crop[0:l, :] = render_blend * render_crop[0:l, :] + depth_blend * depth_crop[0:l, :]
        render_crop[-l:, :] = render_blend * render_crop[-l:, :] + depth_blend * depth_crop[-l:, :]
        render_crop += np.random.normal(0, 1, render_crop.shape)
        depth_crop[render_crop != 0] = render_crop[render_crop != 0]


def colormap(depth, vmin, vmax, min_x, min_y, max_x, max_y, cmap=cv2.COLORMAP_JET):
    cliped = np.clip(depth, vmin, vmax)
    cv2.normalize(cliped, cliped, 1, 0, cv2.NORM_MINMAX)
    im_color = cv2.applyColorMap((cliped[min_y:max_y, min_x:max_x] * 255).astype(np.uint8), cmap)
    return im_color

def compare(root_path, n_with, n_without, coords):
    output_path = os.path.join(root_path, "output")
    rgb = np.array(Image.open(os.path.join(output_path, "{}.png".format(n_with))))

    depth_with = np.array(Image.open(os.path.join(output_path, "{}d.png".format(n_with)))).astype(np.float32)
    depth_without = np.array(Image.open(os.path.join(output_path, "{}d.png".format(n_without)))).astype(np.float32)
    depth_render = np.array(Image.open(os.path.join(output_path, "{}d_render.png".format(n_with)))).astype(np.float32)
    depth_render = cv2.resize(depth_render, depth_with.shape[::-1])

    depth_repaired = depth_with.copy()
    pixel_window = 7
    repair(depth_repaired, depth_render, coords, pixel_window)

    depth_with[depth_render == 0] = 0
    depth_without[depth_render == 0] = 0
    depth_repaired[depth_render == 0] = 0

    error_raw = np.abs(depth_with - depth_without)
    error_fix = np.abs(depth_repaired - depth_without)
    mean_error_raw = np.mean(error_raw[depth_render != 0])
    mean_error_fix = np.mean(error_fix[depth_render != 0])

    raw_errors = []
    fix_errors = []
    for marker_coord in coords:
        raw_crop = error_raw[marker_coord[1] - pixel_window:marker_coord[1] + pixel_window,
                             marker_coord[0] - pixel_window:marker_coord[0] + pixel_window]
        fix_crop = error_fix[marker_coord[1] - pixel_window:marker_coord[1] + pixel_window,
                            marker_coord[0] - pixel_window:marker_coord[0] + pixel_window]
        raw_errors.append(np.mean(np.square(raw_crop)))
        fix_errors.append(np.mean(np.square(fix_crop)))


    print("Raw error : {}".format(math.sqrt(sum(raw_errors)/len(raw_errors))))
    print("Fix error : {}".format(math.sqrt(sum(fix_errors)/len(fix_errors))))

    cmap = "plasma"
    min_x = 870
    max_x = 950
    min_y = 500
    max_y = 580

    vmin = 1100
    vmax = 1230

    error_max = 20

    plt.subplot("241")
    plt.imshow(depth_with[min_y:max_y, min_x:max_x], vmin=vmin, vmax=vmax, cmap="gray")
    plt.subplot("242")
    plt.imshow(depth_without[min_y:max_y, min_x:max_x], vmin=vmin, vmax=vmax, cmap="gray")
    plt.subplot("243")
    plt.imshow(depth_render[min_y:max_y, min_x:max_x], vmin=vmin, vmax=vmax, cmap="gray")
    plt.subplot("244")
    plt.imshow(depth_repaired[min_y:max_y, min_x:max_x], vmin=vmin, vmax=vmax, cmap="gray")

    plt.subplot("245")
    plt.imshow(error_raw[min_y:max_y, min_x:max_x], vmin=0, vmax=error_max, cmap=cmap)
    plt.subplot("246")
    plt.imshow(error_fix[min_y:max_y, min_x:max_x], vmin=0, vmax=error_max, cmap=cmap)
    plt.subplot("247")
    plt.imshow(np.abs(depth_repaired - depth_with)[min_y:max_y, min_x:max_x], vmin=0, vmax=error_max, cmap=cmap)
    plt.subplot("248")
    mpb = plt.imshow(np.abs(depth_without - depth_with)[min_y:max_y, min_x:max_x], vmin=0, vmax=error_max, cmap=cmap)
    plt.colorbar(orientation="vertical")

    diff_error = np.abs(depth_repaired - depth_with)

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.colorbar(mpb, ax=ax)
    ax.remove()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'plot_onlycbar.png'))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth_with = plt.cm.gray(norm(depth_with[min_y:max_y, min_x:max_x]))
    depth_repaired = plt.cm.gray(norm(depth_repaired[min_y:max_y, min_x:max_x]))
    depth_render = plt.cm.gray(norm(depth_render[min_y:max_y, min_x:max_x]))
    depth_without = plt.cm.gray(norm(depth_without[min_y:max_y, min_x:max_x]))


    norm = plt.Normalize(vmin=0, vmax=error_max)
    error_raw = plt.cm.plasma(norm(error_raw[min_y:max_y, min_x:max_x]))
    error_fix = plt.cm.plasma(norm(error_fix[min_y:max_y, min_x:max_x]))
    error_diff = plt.cm.plasma(norm(diff_error[min_y:max_y, min_x:max_x]))

    plt.imsave(os.path.join(output_path, "rgb_with_marker.png"), rgb[min_y:max_y, min_x:max_x, :])
    plt.imsave(os.path.join(output_path, "depth_with_marker.png"), depth_with)
    plt.imsave(os.path.join(output_path, "depth_repaired.png"), depth_repaired)
    plt.imsave(os.path.join(output_path, "depth_render.png"), depth_render)
    plt.imsave(os.path.join(output_path, "depth_without_marker.png"), depth_without)

    plt.imsave(os.path.join(output_path, "error_raw.png"), error_raw)
    plt.imsave(os.path.join(output_path, "error_repaired.png"), error_fix)
    plt.imsave(os.path.join(output_path, "error_diff.png"), error_diff)


    plt.show()


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/marker_compare"
    #capture(root_path)
    coords = np.load(os.path.join(root_path, "output/markers0.npy")).astype(int)
    compare(root_path, 0, 2, coords)

    #depth_with = np.array(Image.open(os.path.join(root_path, "output/0d.png"))).astype(np.float32)
    #plt.imshow(depth_with, vmin=923, vmax=1300)
    #plt.show()

    #np.save(os.path.join(root_path, "output/markers0.npy"), np.array([[930, 508],
    #                                                                  [883, 512],
    #                                                                  [895, 525],
    #                                                                  [935, 533]]))
