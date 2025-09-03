from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


class DeepTrackBinGeoCallback(LoopCallbackBase):
    def __init__(self, file_output_path):
        self.dof_losses = []
        self.class_losses = [[], [], [], [], [], []]

        self.count = 0
        self.file_output_path = file_output_path
        self.translation_bins = None
        self.rotation_bins = None
        self.translation_range = None
        self.rotation_range = None
        self.translation_centers = None
        self.rotation_centers = None

    @staticmethod
    def compute_bin_center(bin, max):
        bin_center = np.zeros(bin.shape)
        for i in range(len(bin)):
            bottom = bin[i]
            if i + 1 == len(bin):
                top = max
            else:
                top = bin[i + 1]
            step = top - bottom
            bin_center[i] = bin[i] + step / 2
        return bin_center

    def batch(self, predictions, network_inputs, targets, isvalid=True):
        if self.translation_bins[0] != -self.translation_range:
            self.translation_bins = np.append(-self.translation_range, self.translation_bins)
        if self.rotation_bins[0] != -self.rotation_range:
            self.rotation_bins = np.append(-self.rotation_range, self.rotation_bins)
        if self.translation_centers is None:
            self.translation_centers = self.compute_bin_center(self.translation_bins, self.translation_range)
        if self.rotation_centers is None:
            self.rotation_centers = self.compute_bin_center(self.rotation_bins, self.rotation_range)

        target = targets[0]
        pose = predictions[0].data
        pose[:, :3] /= self.translation_range
        pose[:, 3:] /= self.rotation_range
        target[:, :3] /= self.translation_range
        target[:, 3:] /= self.rotation_range

        dof_loss = F.mse_loss(Variable(pose), Variable(target)).data[0]
        for i in range(6):
            component_loss = F.mse_loss(Variable(pose[:, i]), Variable(target[:, i])).data[0]
            self.class_losses[i].append(component_loss)
        self.dof_losses.append(dof_loss)

    def epoch(self, loss, data_time, batch_time, isvalid=True):

        losses = [sum(self.dof_losses) / len(self.dof_losses)]
        for class_loss in self.class_losses:
            losses.append(sum(class_loss) / len(class_loss))

        self.console_print(loss, data_time, batch_time, losses, isvalid)
        filename = "validation_data.csv" if isvalid else "training_data.csv"
        self.file_print(os.path.join(self.file_output_path, filename),
                        loss, data_time, batch_time, losses)
        self.dof_losses = []
        self.class_losses = [[], [], [], [], [], []]

    def set_bins(self, translation_bins, rotation_bins):
        self.translation_bins = translation_bins
        self.rotation_bins = rotation_bins

    def set_ranges(self, translation_range, rotation_range):
        self.translation_range = translation_range
        self.rotation_range = rotation_range
