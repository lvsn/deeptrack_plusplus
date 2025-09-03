import numpy as np
import random
import torch

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase


class DeepTrackMaskLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[]):
        super(DeepTrackMaskLoader, self).__init__(root, pretransforms, posttransforms, target_transform)

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        sample = [rgbA, depthA, rgbB, depthB, initial_pose.to_parameters()]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)
        pose_labels[:3] /= float(self.metadata["translation_range"])
        pose_labels[3:] /= float(self.metadata["rotation_range"])
        if self.pretransforms:
            sample = self.pretransforms[0](sample)
        mask = sample[-1]
        if mask is None:
            occluder_mask = np.zeros((depthA.shape), dtype=np.uint8)
            mask = (depthB > occluder_mask)
        if random.randint(0, 1):
            sample[0][:, :, :] *= mask[:, :, np.newaxis]
            sample[1][:, :] *= mask[:, :]
        mask = mask.astype(np.float32)

        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])
        """
        import matplotlib.pyplot as plt
        rgbd = sample[1].numpy()
        plt.subplot(1, 2, 1)
        plt.imshow(rgbd[3, :, :])
        plt.subplot(1, 2, 2)
        plt.imshow(mask.T)
        plt.show()
        """
        return sample, [pose_labels, torch.from_numpy(mask.T)]
