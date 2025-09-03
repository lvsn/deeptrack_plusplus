import numpy as np
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from scipy.stats import cauchy
from scipy.misc import imresize


class DeepTrackBinMaskLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[],
                 n_bin=40, linear_bins=True, callback=None):
        self.n_bin = n_bin
        self.callback = callback
        self.linear_bins = linear_bins
        self.mask_sizes = [(74, 74), (37, 37), (18, 18), (9, 9), (4, 4)]
        super(DeepTrackBinMaskLoader, self).__init__(root, pretransforms, posttransforms, target_transform)
        # compute Bins
        t_max = float(self.metadata["translation_range"])
        t_min = -t_max
        r_max = float(self.metadata["rotation_range"])
        r_min = -r_max
        if self.linear_bins:
            self.translation_bins = self.get_bins(t_min, t_max, self.n_bin)
            self.rotation_bins = self.get_bins(r_min, r_max, self.n_bin)
        else:
            self.translation_bins = self.get_gauss_bins(t_min, t_max, 0.001, n_bin)
            self.rotation_bins = self.get_gauss_bins(r_min, r_max, 0.05, n_bin)
        if callback:
            self.callback.set_bins(self.translation_bins, self.rotation_bins)
            self.callback.set_ranges(t_max, r_max)

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

        binned_labels = np.zeros(pose_labels.shape, dtype=int)
        binned_labels[:3] = np.digitize(pose_labels[:3], self.translation_bins)
        binned_labels[3:] = np.digitize(pose_labels[3:], self.rotation_bins)
        if self.linear_bins:
            binned_labels -= 1

        targets = np.zeros((6, self.n_bin), dtype=np.float32)
        for i, label in enumerate(binned_labels):
            targets[i, label] = 1

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

        return sample, [targets[0], targets[1], targets[2],
                        targets[3], targets[4], targets[5],
                        masks[0].astype(np.float32), masks[1].astype(np.float32),
                        masks[2].astype(np.float32), masks[3].astype(np.float32), masks[4].astype(np.float32),
                        pose_labels]

    @staticmethod
    def get_gauss_bins(minval, maxval, sigma, size):
        """Remember, bin 0 = below value! last bin mean >= maxval"""
        step = (maxval - minval) / size
        large_x = np.linspace(minval, maxval - step, 1000000)
        x = np.linspace(minval, maxval - step, size)
        rv = cauchy(0, sigma)  # gennorm, laplace
        large_pdf = rv.pdf(large_x)
        large_pdf /= large_pdf.sum()

        integral = 0
        division = 1. / size
        intervals = np.zeros(x.shape[0] - 1)
        i = 0
        for val, proba in zip(large_x, large_pdf):
            integral += proba
            if integral > division:
                intervals[i] = val
                integral = 0
                i += 1
        return intervals

    @staticmethod
    def get_bins(minval, maxval, size):
        """Remember, bin 0 = below value! last bin mean >= maxval"""
        step = (maxval - minval)/size
        x = np.linspace(minval, maxval-step, size)
        #x += step/2
        return x