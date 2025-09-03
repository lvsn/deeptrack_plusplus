from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


class DeepTrackGeoCallback(LoopCallbackBase):
    def __init__(self, file_output_path, is_dof_only=True, update_rate=10):
        self.dof_losses = []
        self.dof_norm_losses = []
        self.count = 0
        self.update_rate = update_rate
        self.file_output_path = file_output_path
        self.is_dof_only = is_dof_only
        self.components_losses = [[], [], [], [], [], []]
        self.labels = ["tx", "ty", "tz", "rx", "ry", "rz"]

    def batch(self, predictions, network_inputs, targets, is_train=True, tensorboard_logger=None):
        prediction = predictions[0].data.clone()
        target = targets[0]

        dof_loss = F.mse_loss(Variable(prediction), Variable(target)).data[0]

        t_range = targets[-2]
        r_range = targets[-1]
        prediction[:, :3] /= t_range
        prediction[:, 3:] /= r_range

        target[:, :3] /= t_range
        target[:, 3:] /= r_range
        dof_norm_loss = F.mse_loss(Variable(prediction), Variable(target)).data[0]
        for i in range(6):
            component_loss = F.mse_loss(Variable(prediction[:, i]), Variable(target[:, i])).data[0]
            self.components_losses[i].append(component_loss)

        self.dof_losses.append(dof_loss)
        self.dof_norm_losses.append(dof_norm_loss)

    def epoch(self, epoch, loss, data_time, batch_time, is_train=True, tensorboard_logger=None):

        avr_dof = sum(self.dof_losses)/len(self.dof_losses)
        avr_norm_dof = sum(self.dof_norm_losses)/len(self.dof_norm_losses)
        losses = [avr_dof, avr_norm_dof]
        tensorboard_logger.scalar_summary("mse", avr_dof, epoch + 1, is_train=is_train)
        tensorboard_logger.scalar_summary("norm_mse", avr_norm_dof, epoch + 1, is_train=is_train)
        for label, class_loss in zip(self.labels, self.components_losses):
            avr = sum(class_loss) / len(class_loss)
            losses.append(avr)
            tensorboard_logger.scalar_summary(label, avr, epoch + 1, is_train=is_train)

        self.console_print(loss, data_time, batch_time, losses, is_train)
        filename = "training_data.csv" if is_train else "validation_data.csv"
        self.file_print(os.path.join(self.file_output_path, filename),
                        loss, data_time, batch_time, losses)
        self.dof_losses = []
        self.dof_norm_losses = []
        self.components_losses = [[], [], [], [], [], []]
