import numpy as np
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase


class DeepTrackZLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[]):
        super(DeepTrackZLoader, self).__init__(root, pretransforms, posttransforms, target_transform)

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
        z = -initial_pose.to_parameters()[2]
        w = 1
        if z > 0.5:
            min = 0.5
            max = 1.2
            w = 1 - ((z-min)/(max-min))
        weights = np.ones(6, dtype=np.float32)
        weights[2:5] = w
        return sample, [pose_labels, weights]
