import numpy as np
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from scipy.stats import cauchy
#from scipy.misc import imresize


class DeepTrackCompositionLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[],
                 n_bin=40, linear_bins=True, callback=None):
        self.n_bin = n_bin
        self.callback = callback
        self.linear_bins = linear_bins
        self.mask_sizes = [(74, 74), (37, 37), (18, 18), (9, 9), (4, 4)]
        super(DeepTrackCompositionLoader, self).__init__(root, pretransforms, posttransforms, target_transform)

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        masks = []
        maskA = (depthA != 0).astype(int)
        maskB = (depthB != 0).astype(int)
        mask_object = maskB.copy()
        for size in self.mask_sizes[:2]:
            maskB2 = imresize(maskB, size, interp='nearest')
            maskA2 = imresize(maskA, size, interp='nearest')
            masks.append(maskB2)
        maskAB = np.bitwise_or(maskA2, maskB2)
        for size in self.mask_sizes[2:]:
            maskAB2 = imresize(maskAB, size, interp='nearest')
            masks.append(maskAB2)

        sample = [rgbA, depthA, rgbB, depthB, initial_pose.to_parameters()]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)
        pose_labels[:3] /= float(self.metadata["translation_range"])
        pose_labels[3:] /= float(self.metadata["rotation_range"])

        if self.pretransforms:
            sample = self.pretransforms[0](sample)
        occluder_mask = sample[-1]
        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])

        A, B = sample

        rgbB_mask = B.numpy() * mask_object.T[np.newaxis, :, :]
        if occluder_mask is not None:
            occ_mask = occluder_mask.astype(int)
            rgbB_mask *= occ_mask.T[np.newaxis, :, :]
            for mask in masks:
                occ_mask2 = imresize(occ_mask, mask.shape, interp='nearest')
                mask[:, :] = np.bitwise_and(mask, occ_mask2)
        for mask in masks:
            mask[:, :] = mask.T/255

        sample = [A, B, rgbB_mask.astype(np.float32),
                  masks[0].astype(np.float32), masks[1].astype(np.float32),
                  masks[2].astype(np.float32), masks[3].astype(np.float32), masks[4].astype(np.float32)]

        return sample, [pose_labels,
                        masks[0].astype(np.float32), masks[1].astype(np.float32),
                        masks[2].astype(np.float32), masks[3].astype(np.float32), masks[4].astype(np.float32)]

