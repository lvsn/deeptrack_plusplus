import numpy as np
import math
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import compute_2Dboundingbox
import yaml
import os


class DeepTrackGeoLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[]):
        super(DeepTrackGeoLoader, self).__init__(root, pretransforms, posttransforms, target_transform)
        self.models = yaml.load(open(os.path.join(self.root, "models.yml")))["models"]

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        vertex = self.get_3d_model_from_input(initial_pose, depthA)
        vertex = vertex[:, np.random.choice(vertex.shape[1], 2500, replace=True)]
        initial_pose = initial_pose.to_parameters().astype(np.float32)
        sample = [rgbA, depthA, rgbB, depthB, initial_pose]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)
        translation_range = np.array([float(self.metadata["translation_range"])], dtype=np.float32)
        rotation_range = np.array([float(self.metadata["rotation_range"])], dtype=np.float32)
        if self.pretransforms:
            sample = self.pretransforms[0](sample)
        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])
        K = self.camera.matrix().astype(np.float32)
        K[0, 0] *= -1    # invert x axis (opengl)
        #TODO : is 2500 a good value? could we automate this meta parameter
        return sample, [pose_labels, vertex, initial_pose, K, rgbA, rgbB, translation_range, rotation_range]

    def get_3d_model_from_input(self, pose, depth):
        """
        Compute new camera matrix from the crop/resize process in the dataset generation.
        :param pose:
        :param depth:
        :return:
        """
        new_camera = self.camera.copy()
        bb = compute_2Dboundingbox(pose, self.camera, int(float(self.metadata["bounding_box_width"])), scale=(1000, -1000, -1000))
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        bb_w = right - left
        bb_h = bottom - top
        image_size = int(self.metadata["image_size"])
        new_camera.width = image_size
        new_camera.height = image_size
        new_camera.center_x = image_size / 2.
        new_camera.center_y = image_size / 2.
        fov_x = 2 * math.atan2(self.camera.width, 2 * new_camera.focal_x)
        fov_y = 2 * math.atan2(self.camera.height, 2 * new_camera.focal_y)
        fov_x = fov_x * bb_w / self.camera.width
        fov_y = fov_y * bb_h / self.camera.height
        new_camera.focal_x = new_camera.width / (2 * math.tan(fov_x / 2))
        new_camera.focal_y = new_camera.height / (2 * math.tan(fov_y / 2))

        # back project points
        new_depthA = depth / 1000
        vertex = new_camera.backproject_depth(new_depthA)
        vertex = vertex[vertex.any(axis=1)]  # remove zeros
        vertex[:, 1:] *= -1
        vertex = pose.inverse().dot(vertex)
        return vertex.astype(np.float32).T
