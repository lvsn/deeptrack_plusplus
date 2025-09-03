import os
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
import cv2
import math
from deep_6dof_tracking.data.utils import compute_2Dboundingbox
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler

from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.camera import Camera


if __name__ == '__main__':
    model_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/dragon"
    camera_path = "../data/sensors/camera_parameter_files/synthetic.json"
    image_size = (150, 150)

    camera = Camera.load_from_json(camera_path)
    geometry_path = os.path.join(model_path, "geometry.ply")
    ao_path = os.path.join(model_path, "ao.ply")
    vpRender = ModelRenderer2(geometry_path, "../data/shaders", camera, [image_size])
    vpRender.load_ambiant_occlusion_map(ao_path)

    points = vpRender.data['a_position']
    max_width = maximum_width(points) * 1000

    sphere_sampler = UniformSphereSampler(0.4, 1.2)

    while True:
        random_pose = sphere_sampler.get_random()
        random_transform = sphere_sampler.random_normal_magnitude(0.02, math.radians(20))
        bb = compute_2Dboundingbox(random_pose, camera, max_width, scale=(1000, -1000, -1000))
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        vpRender.setup_camera(camera, left, right, bottom, top)
        rgbA, depthA = vpRender.render_image(random_pose)
        cv2.imshow("test", rgbA[:, :, ::-1])
        cv2.waitKey()




