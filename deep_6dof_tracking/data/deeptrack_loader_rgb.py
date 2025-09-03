import random

import numpy as np
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase


class DeepTrackLoaderRGB(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[]):
        super(DeepTrackLoaderRGB, self).__init__(root, pretransforms, posttransforms, target_transform)

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        sample = [rgbA, depthA, rgbB, depthB, initial_pose.to_parameters()]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)
        pose_labels[:3] /= float(self.metadata["translation_range"])
        pose_labels[3:] /= float(self.metadata["rotation_range"])
        if self.pretransforms:
            sample = self.pretransforms[0](sample)
        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])

        sample[1][3, :, :] = 0

        return sample, [pose_labels]