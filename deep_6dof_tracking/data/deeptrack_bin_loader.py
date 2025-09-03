import numpy as np
from scipy.ndimage import gaussian_filter1d

from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from scipy.stats import norm, cauchy


class DeepTrackBinLoader(DeepTrackLoaderBase):
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[],
                 n_bin=40, linear_bins=True, callback=None, model=None):
        self.n_bin = n_bin
        self.callback = callback
        self.linear_bins = linear_bins
        super(DeepTrackBinLoader, self).__init__(root, pretransforms, posttransforms, target_transform)
        # compute Bins
        t_max = float(self.metadata["translation_range"])
        t_min = -t_max
        r_max = float(self.metadata["rotation_range"])
        r_min = -r_max
        if linear_bins:
            self.translation_bins = self.get_bins(t_min, t_max, self.n_bin)
            self.rotation_bins = self.get_bins(r_min, r_max, self.n_bin)
        else:
            self.translation_bins = self.get_gauss_bins(t_min, t_max, self.meta["translation_variance"], n_bin)
            self.rotation_bins = self.get_gauss_bins(r_min, r_max, self.meta["rotation_variance"], n_bin)

        if callback:
            self.callback.set_bins(self.translation_bins, self.rotation_bins)
            self.callback.set_ranges(t_max, r_max)
        if model:
            model.set_bins(self.translation_bins, self.rotation_bins, t_max, r_max)

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)
        sample = [rgbA, depthA, rgbB, depthB, initial_pose.to_parameters()]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)
        initial_pose = initial_pose.to_parameters().astype(np.float32)
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
        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])
        return sample, [targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], initial_pose, pose_labels]

    @staticmethod
    def smooth_targets(val, sigma=0.5):
        val = gaussian_filter1d(val, sigma=sigma, axis=1, order=0, mode='constant')
        val = gaussian_filter1d(val, sigma=sigma, axis=1, order=0, mode='constant')
        val /= val.sum(axis=1)[:, np.newaxis]
        return val

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    min = -0.1
    max = 0.1
    n_bins = 21
    bins = DeepTrackBinLoader.get_gauss_bins(min, max, 0.01, n_bins)

    #bins = DeepTrackBinLoader.get_bins(min, max, n_bins)
    print(bins)
    Y = np.zeros(len(bins))
    plt.plot(bins, Y, linestyle='--', marker='o', color='b')
    plt.show()

    value = 0.01
    print("Example : {} -> {}".format(value, np.digitize(value, bins)))