import torch
from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


class DeepTrackCallbackDA(LoopCallbackBase):
    def __init__(self, file_output_path, is_dof_only=True, update_rate=10):
        self.dof_losses = []
        self.count = 0
        self.update_rate = update_rate
        self.file_output_path = file_output_path
        self.is_dof_only = is_dof_only
        self.components_losses = [[], [], [], [], [], [], [], [], []]
        self.labels = ["tx", "ty", "tz", "rx", "ry", "rz", "da", "grad_mag", "latent_mag"]

    def batch(self, predictions, network_inputs, targets, is_train=True, tensorboard_logger=None):
        prediction = predictions[0].data.clone()

        prediction_da_real = predictions[1].data
        prediction_da_fake = predictions[2].data

        prediction_da = torch.cat((prediction_da_real, prediction_da_fake))
        
        softmax = F.softmax(prediction_da, dim=1)

        val, max_index = torch.max(softmax, dim=1)
        half = int(prediction_da.size(0) / 2)
        da_targets = np.zeros(prediction_da.size(0))
        da_targets[half:] = 1
        da_targets = torch.from_numpy(da_targets.astype(int)).cuda()

        da_accuracy = float((max_index == da_targets).sum()) / len(max_index)

        grad_magnitude = 0
        if predictions[3] is not None:
            grad_magnitude = float(predictions[3])

        latent_grad_magnitude = 0
        if predictions[4] is not None:
            latent_grad_magnitude = float(predictions[4])

        dof_loss = F.mse_loss(Variable(prediction), Variable(targets[0])).item()
        for i in range(6):
            component_loss = F.mse_loss(Variable(prediction[:, i]), Variable(targets[0][:, i])).item()
            self.components_losses[i].append(component_loss)
        self.components_losses[6].append(da_accuracy)
        self.components_losses[7].append(grad_magnitude)
        self.components_losses[8].append(latent_grad_magnitude)
        self.dof_losses.append(dof_loss)

    def epoch(self, epoch, loss, data_time, batch_time, is_train=True, tensorboard_logger=None):

        losses = [sum(self.dof_losses) / len(self.dof_losses)]
        for label, class_loss in zip(self.labels, self.components_losses):
            avr = sum(class_loss) / len(class_loss)
            losses.append(avr)
            if tensorboard_logger:
                tensorboard_logger.scalar_summary(label, avr, epoch + 1, is_train=is_train)

        self.console_print(loss, data_time, batch_time, losses, is_train)
        filename = "training_data.csv" if is_train else "validation_data.csv"
        self.file_print(os.path.join(self.file_output_path, filename),
                        loss, data_time, batch_time, losses)
        self.dof_losses = []
        self.components_losses = [[], [], [], [], [], [], [], [], []]
