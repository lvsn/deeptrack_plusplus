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

    color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height # Default: 1920, 1080
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