import math
import numpy as np
import torch

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import compute_2Dboundingbox, combine_view_transform
from deep_6dof_tracking.utils.plyparser import PlyParser


class DeepTrackGeoFlowLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[]):
        super(DeepTrackGeoFlowLoader, self).__init__(root, pretransforms, posttransforms, target_transform)

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        final_pose = combine_view_transform(initial_pose, transformed_pose)

        vertexA, zoomed_camera = self.get_3d_model_from_input(initial_pose, depthA)
        vertexB, zoomed_camera = self.get_3d_model_from_input(final_pose, depthB)
        # Setup 3D model for projection loss
        vertex_model = vertexA.astype(np.float32).T
        vertex_model = vertex_model[:, np.random.choice(vertex_model.shape[1], 2500, replace=False)]

        # Setup optical flow target
        vertexA = initial_pose.dot(vertexB)
        vertexB = final_pose.dot(vertexB)
        vertexA[:, 1:] *= -1
        vertexB[:, 1:] *= -1

        pointsA_2D = zoomed_camera.project_points(vertexA, round=False)
        pointsB_2D = zoomed_camera.project_points(vertexB, round=False)
        disparityB = pointsB_2D - pointsA_2D
        optical_flow = self.compute_disparity(pointsB_2D, zoomed_camera, disparityB)


        initial_pose = initial_pose.to_parameters().astype(np.float32)
        sample = [rgbA, depthA, rgbB, depthB, initial_pose]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)
        translation_range = np.array([float(self.metadata["translation_range"])], dtype=np.float32)
        rotation_range = np.array([float(self.metadata["rotation_range"])], dtype=np.float32)
        if self.pretransforms:
            sample = self.pretransforms[0](sample)
        if sample[-1] is not None:
            # apply occluder mask over optical flow
            optical_flow = optical_flow * sample[-1][:, :, np.newaxis]
        """
        import matplotlib.pyplot as plt
        plt.subplot("221")
        plt.imshow(depthA)
        plt.subplot("222")
        plt.imshow(depthB)
        plt.subplot("223")
        plt.imshow(depthB_rec[:, :, 0])
        plt.subplot("224")
        plt.imshow(depthB_rec[:, :, 1])
        plt.show()
        """
        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])
        K = self.camera.matrix().astype(np.float32)
        K[0, 0] *= -1    # invert x axis (opengl)

        #normalize optical flow for tanh output
        optical_flow /= (int(self.metadata["image_size"])/2)
        return sample, [pose_labels, vertex_model, initial_pose, K, optical_flow.astype(np.float32).T,
                        #sample[0][3, :, :], sample[1][3, :, :],
                        translation_range, rotation_range]

    def get_3d_model_from_input(self, pose, depth):
        """
        Compute new camera matrix from the crop/resize process in the dataset generation.
        :param pose:
        :param depth:
        :return:
        """
        new_camera = self.camera.copy()
        #TODO : remove this structure from metadata..
        bb = compute_2Dboundingbox(pose, self.camera, int(self.metadata["object_width"]["dragon"]), scale=(1000, -1000, -1000))
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
        vertex = vertex[vertex.any(axis=1)]
        vertex[:, 1:] *= -1
        vertex = pose.inverse().dot(vertex)
        return vertex, new_camera

    @staticmethod
    def compute_disparity(points, camera, disparity):
        points = points.astype(int)
        points[:, 0] = points[:, 0].clip(0, camera.height - 1)
        points[:, 1] = points[:, 1].clip(0, camera.width - 1)
        disparity_map = np.zeros((camera.height, camera.width, 2))
        disparity_map[points[:, 0], points[:, 1], 0] = disparity[:, 0]
        disparity_map[points[:, 0], points[:, 1], 1] = disparity[:, 1]
        return disparity_map