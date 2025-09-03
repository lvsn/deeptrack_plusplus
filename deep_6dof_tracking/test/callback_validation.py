from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch


class DeepTrackBinCallbackValidation(LoopCallbackBase):
    def __init__(self):
        self.dof_losses = []
        self.class_losses = [[], [], [], [], [], []]

        self.count = 0
        self.translation_bins = None
        self.rotation_bins = None
        self.translation_range = None
        self.rotation_range = None
        self.bins = 20
        self.z_bins = np.linspace(0.4, 1.2, self.bins)
        self.z_distributions = []
        for i in range(self.bins):
            self.z_distributions.append([])
        self.epoch_loss = []

    def batch(self, predictions, network_inputs, targets, isvalid=True):
        pose = np.zeros((predictions[0].size(0), 6))
        if self.translation_bins[0] != -self.translation_range:
            self.translation_bins = np.append(-self.translation_range, self.translation_bins)
        if self.rotation_bins[0] != -self.rotation_range:
            self.rotation_bins = np.append(-self.rotation_range, self.rotation_bins)
        for i in range(6):
            value, argmax = torch.max(predictions[i].data, 1)
            argmax = argmax.cpu().numpy()
            # ugly, simply retrieve the pose from the classes
            if i < 3:
                bottom = self.translation_bins[argmax]
                top = bottom.copy()
                top_index = argmax == self.translation_bins.shape[0] - 1
                not_top_index = argmax != self.translation_bins.shape[0] - 1
                top[not_top_index] = self.translation_bins[argmax[not_top_index] + 1]
                top[top_index] = self.translation_range
                step = top - bottom
                pose[:, i] = self.translation_bins[argmax] + step / 2
            else:
                bottom = self.rotation_bins[argmax]
                top = bottom.copy()
                top_index = argmax == self.rotation_bins.shape[0] - 1
                not_top_index = argmax != self.translation_bins.shape[0] - 1
                top[top_index] = self.rotation_range
                top[not_top_index] = self.translation_bins[argmax[not_top_index] + 1]
                step = top - bottom
                pose[:, i] = self.rotation_bins[argmax] + step / 2
        pose = torch.from_numpy(pose).type(predictions[0].data.type())
        target = targets[-1]
        init_pose = targets[-2].cpu().numpy()

        errors = np.abs((pose - target).cpu().numpy())[:, 2]

        pose[:, :3] /= self.translation_range
        pose[:, 3:] /= self.rotation_range
        target[:, :3] /= self.translation_range
        target[:, 3:] /= self.rotation_range

        dof_loss = F.mse_loss(Variable(pose), Variable(target)).data[0]

        z_distance = np.abs(init_pose[:, 2])
        z_indexes = np.digitize(z_distance, self.z_bins) - 1
        for i in range(len(self.z_distributions)):
            for val in errors[z_indexes == i]:
                self.z_distributions[i].append(val)
        for i in range(6):
            prediction = predictions[i].data.clone()
            l = F.kl_div(prediction, Variable(targets[i])).data[0]
            self.class_losses[i].append(l)
        self.dof_losses.append(dof_loss)

    def epoch(self, loss, data_time, batch_time, isvalid=True):

        self.epoch_loss.append(sum(self.dof_losses) / len(self.dof_losses))

        plt.subplot("211")
        plt.violinplot(self.z_distributions[:-1], self.z_bins[:-1], widths=0.04, showmeans=True, showextrema=True, showmedians=True)
        plt.subplot("212")
        plt.plot(np.arange(len(self.epoch_loss)), self.epoch_loss)
        plt.show()

        losses = [sum(self.dof_losses) / len(self.dof_losses)]
        for class_loss in self.class_losses:
            losses.append(sum(class_loss) / len(class_loss))

        filename = "validation_data.csv" if isvalid else "training_data.csv"
        self.file_print(filename, loss, data_time, batch_time, losses)
        self.dof_losses = []
        self.class_losses = [[], [], [], [], [], []]
        self.z_distributions = []
        for i in range(self.bins):
            self.z_distributions.append([])

    def set_bins(self, translation_bins, rotation_bins):
        self.translation_bins = translation_bins
        self.rotation_bins = rotation_bins

    def set_ranges(self, translation_range, rotation_range):
        self.translation_range = translation_range
        self.rotation_range = rotation_range
