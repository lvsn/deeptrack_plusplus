import numpy as np
import torch
from deep_6dof_tracking.experimental.sphere import Sphere
from torch.autograd import Variable
import matplotlib.pyplot as plt

DEBUG = True


# from https://medium.com/@awildtaber/building-a-rendering-engine-in-tensorflow-262438b2e062

def plane(a, b, c, m):
    def expression(x, y, z):
        return x * a + y * b + z * c + m

    return expression


def sphere(r):
    def expression(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2) - r
    return expression


def union(f, g):
    # f and g are functions
    def unioned(x, y, z):
        return max(f(x, y, z), g(x, y, z))

    return unioned


def intersection(f, g):
    def intersected(x, y, z):
        return min(f(x, y, z), g(x, y, z))

    return intersected


def negation(f):
    def negated(x, y, z):
        return -f(x, y, z)

    return negated


def translate(a, b, c):
    def translated(x, y, z):
        return x - a, y - b, z - c

    return translated


def normalize_vector(vector):
    return vector / np.sqrt(np.sum(np.square(vector), axis=0))


def vector_fill(shape, vector):
    vectors = np.zeros((3, *shape))
    vectors[0, :, :].fill(vector[0])
    vectors[1, :, :].fill(vector[1])
    vectors[2, :, :].fill(vector[2])
    return vectors


if __name__ == '__main__':
    resolution = (500, 500)
    aspect_ratio = resolution[0] / resolution[1]
    min_bounds, max_bounds = (-aspect_ratio, -1), (aspect_ratio, 1)
    resolutions = list(map(lambda x: x * 1j, resolution))
    image_plane_coords = np.mgrid[min_bounds[0]:max_bounds[0]:resolutions[0],
                         min_bounds[1]:max_bounds[1]:resolutions[1]]

    # Find the center of the image plane
    camera_position = np.array([0, -5, 0])
    lookAt = (0, 0, 0)
    camera = camera_position - np.array(lookAt)
    camera_direction = normalize_vector(camera)
    focal_length = 1
    eye = camera + focal_length * camera_direction

    # Coerce into correct shape
    image_plane_center = vector_fill(resolution, camera_position)

    # Convert u,v parameters to x,y,z coordinates for the image plane
    v_unit = [0, 0, -1]
    u_unit = np.cross(camera_direction, v_unit)
    image_plane = image_plane_center + image_plane_coords[0] * vector_fill(resolution, u_unit) + image_plane_coords[
                                                                                                     1] * vector_fill(
        resolution, v_unit)
    # Populate the image plane with initial unit ray vectors
    initial_vectors = image_plane - vector_fill(resolution, eye)
    ray_vectors = normalize_vector(initial_vectors)

    t = np.zeros(resolution)
    space = (ray_vectors * t) + image_plane
    plt.imshow(space.T)
    plt.show()

    x = Variable(torch.from_numpy(space[0, :, :].astype(np.float32)), requires_grad=True)
    y = Variable(torch.from_numpy(space[1, :, :].astype(np.float32)), requires_grad=True)
    z = Variable(torch.from_numpy(space[2, :, :].astype(np.float32)), requires_grad=True)

    func = Sphere(1)
    evaluated_function = func(x, y, z)
    evaluated_function.sum().backward()
    evaluated_grad = evaluated_function.grad
    gradients = np.zeros((3, *resolution), dtype=np.float32)
    gradients[0, :, :] = x.grad.data.cpu().numpy()
    gradients[1, :, :] = y.grad.data.cpu().numpy()
    gradients[2, :, :] = z.grad.data.cpu().numpy()

    if DEBUG:
        plt.subplot("131")
        plt.imshow(gradients[0, :, :])
        plt.subplot("132")
        plt.imshow(gradients[1, :, :])
        plt.subplot("133")
        plt.imshow(gradients[2, :, :])
        plt.show()

    evaluated_function = evaluated_function.data.cpu().numpy()
    plt.imshow(evaluated_function.T)
    plt.show()

    # Iteration operation
    for i in range(50):
        epsilon = 0.0001
        distance = np.abs(evaluated_function)
        distance_step = t - (np.sign(evaluated_function) * np.maximum(distance, epsilon))
        t = distance_step

    # print(np.max(distance_step))
    # print(t)
    # ray_step = t.assign(distance_step)

    light = {"position": np.array([0, 5, 5]), "color": np.array([255, 255, 255])}
    normal_vector = normalize_vector(gradients)
    incidence = normal_vector - vector_fill(resolution, light["position"])
    normalized_incidence = normalize_vector(incidence)
    incidence_angle = np.sum(normalized_incidence * normal_vector, axis=0)

    # Split the color into three channels
    light_intensity = vector_fill(resolution, light['color']) * incidence_angle

    # Add ambient light
    ambient_color = [165, 139, 119]
    with_ambient = light_intensity * 0.5 + vector_fill(resolution, ambient_color) * 0.5
    lighted = with_ambient

    # Mask out pixels not on the surface
    epsilon = 0.0001
    bitmask = distance <= epsilon
    masked = lighted * bitmask.astype(np.float32)
    sky_color = [180, 130, 70]
    background = vector_fill(resolution, sky_color) * np.logical_not(bitmask).astype(np.float32)
    image_data = masked + background

    if DEBUG:
        plt.imshow(image_data.T)
        plt.show()
