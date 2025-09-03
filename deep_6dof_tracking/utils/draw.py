import numpy as np
import cv2


def draw_color_blend(foreground, background):
    """
    Uses pixel 0 to compute blending mask
    :param foreground:
    :param background:
    :return:
    """
    if len(foreground.shape) == 2:
        mask = foreground[:, :] == 0
    else:
        mask = foreground[:, :, 0] == 0
        mask = mask[:, :, np.newaxis]
    return background * mask + foreground


def blend_depth_error(depth_foreground, depth_background, background_rgb):
    foreground_mask = depth_foreground == 0
    depth_background[foreground_mask] = 0

    depth_diff = np.abs(depth_foreground.astype(np.float32) - depth_background.astype(np.float32))
    depth_diff = np.clip(depth_diff, 0, 20)
    #import matplotlib.pyplot as plt
    #plt.imshow(depth_diff)
    #plt.show()
    depth_diff = (depth_diff / np.max(depth_diff) * 255).astype(np.uint8)
    depth_diff = cv2.applyColorMap(depth_diff, cv2.COLORMAP_JET)[:, :, ::-1]
    depth_diff[foreground_mask] = 0
    return background_rgb * foreground_mask[:, :, np.newaxis] + depth_diff


def draw_axis(img, pose, camera, brightness=255):
    points = np.ndarray((4, 3), dtype=float)
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2] = [0, 1, 0]
    points[3] = [0, 0, 1]
    points *= 0.1
    camera_points = pose.dot(points)
    camera_points[:, 0] *= -1
    axis = camera.project_points(camera_points).astype(np.int32)

    cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[1, ::-1]), (0, 0, brightness), 3)
    cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[2, ::-1]), (0, brightness, 0), 3)
    cv2.line(img, tuple(axis[0, ::-1]), tuple(axis[3, ::-1]), (brightness, 0, 0), 3)