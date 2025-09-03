from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


class DeepTrackFlowCallback(LoopCallbackBase):
    def __init__(self, file_output_path, is_dof_only=True, update_rate=10):
        self.dof_losses = []
        self.mask_losses = []
        self.count = 0
        self.update_rate = update_rate
        self.file_output_path = file_output_path
        self.is_dof_only = is_dof_only
        self.components_losses = [[], [], [], [], [], []]

    def batch(self, predictions, network_inputs, targets, is_train=True, tensorboard_logger=None):
        prediction = predictions[0].data.clone()
        target = targets[0]

        dof_loss = F.mse_loss(Variable(prediction), Variable(target)).data[0]
        for i in range(6):
            component_loss = F.mse_loss(Variable(prediction[:, i]), Variable(target[:, i])).data[0]
            self.components_losses[i].append(component_loss)
        mask_loss = 0
        if len(predictions) == 2:
            mask_loss = F.mse_loss(Variable(predictions[1].data.clone()), Variable(targets[1])).data[0]

        self.mask_losses.append(mask_loss)
        self.dof_losses.append(dof_loss)

    def epoch(self, epoch, loss, data_time, batch_time, is_train=True, tensorboard_logger=None):

        losses = [sum(self.dof_losses)/len(self.dof_losses),
                  sum(self.mask_losses)/len(self.mask_losses)]
        for class_loss in self.components_losses:
            losses.append(sum(class_loss) / len(class_loss))

        self.console_print(loss, data_time, batch_time, losses, is_train)
        filename = "validation_data.csv" if is_train else "training_data.csv"
        self.file_print(os.path.join(self.file_output_path, filename),
                        loss, data_time, batch_time, losses)
        self.dof_losses = []
        self.mask_losses = []
        self.components_losses = [[], [], [], [], [], []]
