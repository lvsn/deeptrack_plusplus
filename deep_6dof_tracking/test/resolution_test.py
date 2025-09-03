from deep_6dof_tracking.utils.camera import Camera
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    camera_path = "/home/mathieu/source/deep_6dof_tracking/deep_6dof_tracking/data/sensors/camera_parameter_files/Kinect2_lab.json"
    object_size = 0.20

    camera = Camera.load_from_json(camera_path)
    print("Using camera : {}".format(camera))
    print("Object size : {}".format(object_size))

    X = np.arange(0.5, 2, 0.05)
    Y = np.zeros(X.shape)
    for i, z_distance in enumerate(X):

        points = np.zeros((2, 3))
        points[0, :] = [-object_size/2, 0, z_distance]
        points[1, :] = [object_size/2, 0, z_distance]

        coords = camera.project_points(points)
        Y[i] = coords[1, 1] - coords[0, 1]

    plt.plot(X, Y)
    plt.xlabel("Z distance")
    plt.ylabel("pixel_size")
    plt.show()