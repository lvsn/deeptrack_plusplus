import math

import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase

from deep_6dof_tracking.networks.da_module import GradientRecordHook, GradientScale
from deep_6dof_tracking.networks.fire_module import Fire
import numpy as np
import random


def hook(grad):
    print(grad)

class DomainAdaptation(torch.nn.Module):

    def __init__(self, input_dims):
        super(DomainAdaptation, self).__init__()
        self.always_turn_off_module_train = False
        self.latent = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ELU(inplace=True),
        )

        self.dropout1 = nn.Dropout(0.5)
        self.classifier = nn.Linear(64, 2)

        self.regressor = nn.Linear(64, 6)

        self.grad_scale = GradientScale()
        # gradient hooks
        self.grad_record = GradientRecordHook()

    def forward(self, latent_vector, lambdar):
        x = latent_vector
        x = self.grad_record(x)
        x = GradientScale.apply(x, lambdar)  # check the magnitude after converged to balance the gradient, x10 seems better to get .5, keep it


        x = self.latent(x)
        discr = self.dropout1(x)
        discr = self.classifier(discr)
        soft = F.log_softmax(discr, dim=1)

        return soft, discr


class DeepTrackResNetSplit(NetworkBase):
    def __init__(self, image_size=150, phase=0):
        super(DeepTrackResNetSplit, self).__init__()
        self.phase = phase
        filter_size_1 = 64
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA_rgb = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA_rgb = nn.BatchNorm2d(filter_size_1)
        self.fireA_rgb = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2_rgb = nn.BatchNorm2d(filter_size_1 * 2)

        self.convB_rgb = nn.Conv2d(3, filter_size_1, 3, 2)
        self.batchB_rgb = nn.BatchNorm2d(filter_size_1)
        self.fireB_rgb = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2_rgb = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1_rgb = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1_rgb = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire2_rgb = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2_rgb = nn.BatchNorm2d(filter_size_1 * 16)
        self.fire3_rgb = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3_rgb = nn.BatchNorm2d(filter_size_1 * 24)

        view_width = int(int(int(int(int((image_size - 2)/2)/2)/2)/2)/2)

        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1_rgb = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1_rgb = nn.BatchNorm1d(embedding_size)
        self.fc2_rgb = nn.Linear(embedding_size, 6)

        self.dropout1_rgb = nn.Dropout(0.5)
        self.dropout_AB0_rgb = nn.Dropout2d(0.3)
        self.dropout_AB1_rgb = nn.Dropout2d(0.3)

        # Depth part
        self.convA_depth = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA_depth = nn.BatchNorm2d(filter_size_1)
        self.fireA_depth = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2_depth = nn.BatchNorm2d(filter_size_1 * 2)

        self.convB_depth = nn.Conv2d(3, filter_size_1, 3, 2)
        self.batchB_depth = nn.BatchNorm2d(filter_size_1)
        self.fireB_depth = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2_depth = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1_depth = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1_depth = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire2_depth = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2_depth = nn.BatchNorm2d(filter_size_1 * 16)
        self.fire3_depth = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3_depth = nn.BatchNorm2d(filter_size_1 * 24)

        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1_depth = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1_depth = nn.BatchNorm1d(embedding_size)
        self.fc2_depth = nn.Linear(embedding_size, 6)

        self.dropout1_depth = nn.Dropout(0.5)
        self.dropout_AB0_depth = nn.Dropout2d(0.3)
        self.dropout_AB1_depth = nn.Dropout2d(0.3)

        # Halucinate part
        self.convA_hal = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA_hal = nn.BatchNorm2d(filter_size_1)
        self.fireA_hal = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2_hal = nn.BatchNorm2d(filter_size_1 * 2)

        self.convB_hal = nn.Conv2d(3, filter_size_1, 3, 2)
        self.batchB_hal = nn.BatchNorm2d(filter_size_1)
        self.fireB_hal = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2_hal = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1_hal = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1_hal = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire2_hal = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2_hal = nn.BatchNorm2d(filter_size_1 * 16)
        self.fire3_hal = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3_halh = nn.BatchNorm2d(filter_size_1 * 24)

        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1_hal = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1_hal = nn.BatchNorm1d(embedding_size)
        self.fc2_hal = nn.Linear(embedding_size, 6)

        self.dropout1_hal = nn.Dropout(0.5)
        self.dropout_AB0_hal = nn.Dropout2d(0.3)
        self.dropout_AB1_hal = nn.Dropout2d(0.3)

        self.DA = DomainAdaptation(embedding_size)
        self.record_hook = GradientRecordHook()
        self.count = 0

    def load_hal(self):
        self.convA_hal = copy.deepcopy(self.convA_depth)
        self.batchA_hal = copy.deepcopy(self.batchA_depth)
        self.fireA_hal = copy.deepcopy(self.fireA_depth)
        self.batchA2_hal = copy.deepcopy(self.batchA2_depth)

        self.convB_hal = copy.deepcopy(self.convB_depth)
        self.batchB_hal = copy.deepcopy(self.batchB_depth)
        self.fireB_hal = copy.deepcopy(self.fireB_depth)
        self.batchB2_hal = copy.deepcopy(self.batchB2_depth)

        self.fire1_hal = copy.deepcopy(self.fire1_depth)
        self.batch1_hal = copy.deepcopy(self.batch1_depth)
        self.fire2_hal = copy.deepcopy(self.fire2_depth)
        self.batch2_hal = copy.deepcopy(self.batch2_depth)
        self.fire3_hal = copy.deepcopy(self.fire3_depth)
        self.batch3_halh = copy.deepcopy(self.batch3_depth)

        self.fc1_hal = copy.deepcopy(self.fc1_depth)
        self.fc_bn1_hal = copy.deepcopy(self.fc_bn1_depth)
        self.fc2_hal = copy.deepcopy(self.fc2_depth)

        self.dropout1_hal = copy.deepcopy(self.dropout1_depth)
        self.dropout_AB0_hal = copy.deepcopy(self.dropout_AB0_depth)
        self.dropout_AB1_hal = copy.deepcopy(self.dropout_AB1_depth)

        self.convA_depth.requires_grad = False
        self.batchA_depth.requires_grad = False
        self.fireA_depth.requires_grad = False
        self.batchA2_depth.requires_grad = False

        self.convB_depth.requires_grad = False
        self.batchB_depth.requires_grad = False
        self.fireB_depth.requires_grad = False
        self.batchB2_depth.requires_grad = False

        self.fire1_depth.requires_grad = False
        self.batch1_depth.requires_grad = False
        self.fire2_depth.requires_grad = False
        self.batch2_depth.requires_grad = False
        self.fire3_depth.requires_grad = False
        self.batch3_depth.requires_grad = False

        self.fc1_depth.requires_grad = False
        self.fc_bn1_depth.requires_grad = False
        self.fc2_depth.requires_grad = False

        self.dropout1_depth.requires_grad = False
        self.dropout_AB0_depth.requires_grad = False
        self.dropout_AB1_depth.requires_grad = False

    def forward(self, A, B):
        B_rgb = B[:, :3, :, :]
        B_depth = B[:, 3, :, :].unsqueeze(1)
        B_depth = torch.cat((B_depth, B_depth, B_depth), dim=1)

        if self.phase == 0:
            rgb, _ = self.stream_rgb(A, B_rgb)
            depth, _ = self.stream_depth(A, B_depth)
            return rgb, depth
        elif self.phase == 1:
            hal, hal_latent = self.stream_hal(A, B_rgb)
            depth, depth_latent = self.stream_depth(A, B_depth)
            #depth_latent = depth_latent.detach()

            #self.count += len(A)
            #epoch = self.count / 200000
            # print(epoch)
            #max_epoch = 20
            #p = float(epoch) / max_epoch
            #lambda_slow = 2. / (1. + math.exp(-8. * p)) - 1
            lambda_slow = torch.FloatTensor([1.]).cuda()
            da_hal, da_hal_logit = self.DA(hal_latent, lambda_slow)
            da_depth, da_depth_logit = self.DA(depth_latent, lambda_slow)

            return hal, da_depth_logit, da_hal_logit, self.DA.grad_record.mag, None

    def stream_rgb(self, A, B):
        A = self.batchA_rgb(F.elu(self.convA_rgb(A), inplace=True))
        A = self.batchA2_rgb(F.max_pool2d(torch.cat((self.fireA_rgb(A), A), 1), 2))

        B = self.batchB_rgb(F.elu(self.convB_rgb(B), inplace=True))
        B = self.batchB2_rgb(F.max_pool2d(torch.cat((self.fireB_rgb(B), B), 1), 2))

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0_rgb(self.batch1_rgb(F.max_pool2d(torch.cat((self.fire1_rgb(AB), AB), 1), 2)))
        AB = self.dropout_AB1_rgb(self.batch2_rgb(F.max_pool2d(torch.cat((self.fire2_rgb(AB), AB), 1), 2)))
        AB = self.batch3_rgb(F.max_pool2d(torch.cat((self.fire3_rgb(AB), AB), 1), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1_rgb(AB)
        latent = self.fc_bn1_rgb(F.elu(self.fc1_rgb(AB)))
        rgb = torch.tanh(self.fc2_rgb(latent))
        return rgb, latent

    def stream_depth(self, A, B):
        A = self.batchA_depth(F.elu(self.convA_depth(A), inplace=True))
        A = self.batchA2_depth(F.max_pool2d(torch.cat((self.fireA_depth(A), A), 1), 2))

        B = self.batchB_depth(F.elu(self.convB_depth(B), inplace=True))
        B = self.batchB2_depth(F.max_pool2d(torch.cat((self.fireB_depth(B), B), 1), 2))

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0_depth(self.batch1_depth(F.max_pool2d(torch.cat((self.fire1_depth(AB), AB), 1), 2)))
        AB = self.dropout_AB1_depth(self.batch2_depth(F.max_pool2d(torch.cat((self.fire2_depth(AB), AB), 1), 2)))
        AB = self.batch3_depth(F.max_pool2d(torch.cat((self.fire3_depth(AB), AB), 1), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1_depth(AB)
        latent = self.fc_bn1_depth(F.elu(self.fc1_depth(AB)))
        depth = torch.tanh(self.fc2_depth(latent))
        return depth, latent

    def stream_hal(self, A, B):
        A = self.batchA_hal(F.elu(self.convA_hal(A), inplace=True))
        A = self.batchA2_hal(F.max_pool2d(torch.cat((self.fireA_hal(A), A), 1), 2))

        B = self.batchB_hal(F.elu(self.convB_hal(B), inplace=True))
        B = self.batchB2_hal(F.max_pool2d(torch.cat((self.fireB_hal(B), B), 1), 2))

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0_hal(self.batch1_hal(F.max_pool2d(torch.cat((self.fire1_hal(AB), AB), 1), 2)))
        AB = self.dropout_AB1_hal(self.batch2_hal(F.max_pool2d(torch.cat((self.fire2_hal(AB), AB), 1), 2)))
        AB = self.batch3_halh(F.max_pool2d(torch.cat((self.fire3_hal(AB), AB), 1), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1_hal(AB)
        latent = self.fc_bn1_hal(F.elu(self.fc1_hal(AB)))
        hal = torch.tanh(self.fc2_hal(latent))
        return hal, latent

    def stream(self, A, B, batch_norms):
        B_rgb = B[:, :3, :, :]
        if self.phase == 0:
            rgb, _ = self.stream_rgb(A, B_rgb)
            return rgb, None
        elif self.phase == 1:
            hal, hal_latent = self.stream_hal(A, B_rgb)
            rgb, rgb_latent = self.stream_rgb(A, B_rgb)
            lambda_slow = torch.FloatTensor([0]).cuda()
            #da_soft, da_logit, task_hal = self.DA(hal_latent, lambda_slow)
            #da_soft, da_logit, task_depth = self.DA(depth_latent, lambda_slow)
            return (hal + rgb)/2, None


    def loss(self, predictions, targets):
        if self.phase == 0:
            loss = nn.MSELoss()(predictions[0], targets[0]) + nn.MSELoss()(predictions[1], targets[0])
        elif self.phase == 1:

            da = predictions[1]
            da_ = predictions[2]
            
            half = int(da.size(0))
            da_targets = np.zeros(half*2)
            da_targets[half:] = 1
            # flips the label
            if random.uniform(0, 1) < 0.2:
                da_targets = np.abs(da_targets - 1)
            da_targets = torch.from_numpy(da_targets.astype(int)).cuda()
            da_preds = torch.cat((da, da_), dim=0)

            da_loss = F.nll_loss(F.log_softmax(da_preds, dim=1), da_targets)

            task = nn.MSELoss()(predictions[0], targets[0])

            loss = da_loss*0.001

        return loss
