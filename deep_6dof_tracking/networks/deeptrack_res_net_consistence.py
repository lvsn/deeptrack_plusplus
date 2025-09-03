import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase

from deep_6dof_tracking.networks.da_module import DomainAdaptation, GradientRecordHook
from deep_6dof_tracking.networks.fire_module import Fire
import random
import numpy as np


def hook(grad):
    print(grad)


class DeepTrackConsistence(NetworkBase):
    def __init__(self, image_size=150, phase=None):
        super(DeepTrackConsistence, self).__init__()

        filter_size_1 = 64
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.fireA = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.fireB = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)

        self.fire1 = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.fire2 = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.fire3 = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)

        self.batch_norms = self.generate_batchnorms(filter_size_1, embedding_size)

        view_width = int(int(int(int(int((image_size - 2)/2)/2)/2)/2)/2)

        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

        self.DA = DomainAdaptation(embedding_size)
        self.record_hook = GradientRecordHook()
        self.count = 0

    def generate_batchnorms(self, filter_size, embeding_size):
        batchA = nn.BatchNorm2d(filter_size)
        batchA2 = nn.BatchNorm2d(filter_size * 2)
        batchB = nn.BatchNorm2d(filter_size)
        batchB2 = nn.BatchNorm2d(filter_size * 2)
        batch1 = nn.BatchNorm2d(filter_size * 8)
        batch2 = nn.BatchNorm2d(filter_size * 16)
        batch3 = nn.BatchNorm2d(filter_size * 24)
        fc_bn1 = nn.BatchNorm1d(embeding_size)
        return nn.ModuleList([batchA, batchA2, batchB, batchB2, batch1, batch2, batch3, fc_bn1])

    def forward(self, A, B):

        # Having a different minibatch for each stream seems to be a problem a test time...
        # The network might get more robust when it sees all the case at thes time (well when the batch norm sees all of them..)

        minibatch_half = int(len(B)/2)
        B[minibatch_half:, 3, :, :] = 0

        out, latent = self.stream(A, B, None)
        #depth_out, depth_latent = self.stream(A.clone(), B_depth, self.depth_batch_norms)

        self.count += len(A)
        epoch = self.count / 200000
        #print(epoch)
        max_epoch = 20
        p = float(epoch) / max_epoch
        lambda_slow = 2. / (1. + math.exp(-8. * p)) - 1
        lambda_slow = torch.FloatTensor([lambda_slow*13]).cuda()

        da_soft, da_logit = self.DA(latent, lambda_slow)

        return out, da_logit[:minibatch_half, :], da_logit[minibatch_half:], self.DA.grad_record.mag, self.record_hook.mag

    def stream(self, A, B, batch_norms):
        A = self.batch_norms[0](F.elu(self.convA(A), inplace=True))
        A = self.batch_norms[1](F.max_pool2d(torch.cat((self.fireA(A), A), 1), 2))

        B = self.batch_norms[2](F.elu(self.convB(B), inplace=True))
        B = self.batch_norms[3](F.max_pool2d(torch.cat((self.fireB(B), B), 1), 2))

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0(self.batch_norms[4](F.max_pool2d(torch.cat((self.fire1(AB), AB), 1), 2)))
        AB = self.dropout_AB1(self.batch_norms[5](F.max_pool2d(torch.cat((self.fire2(AB), AB), 1), 2)))
        AB = self.batch_norms[6](F.max_pool2d(torch.cat((self.fire3(AB), AB), 1), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = self.batch_norms[7](F.elu(self.fc1(AB)))
        #AB = self.record_hook(AB)
        latent = AB
        out = torch.tanh(self.fc2(latent))
        return out, latent

    def loss(self, predictions, targets):
        task_loss = nn.MSELoss()(predictions[0], targets[0])
        # rgb is considered as fake, so it has to be predicted as 0
        # GANs needs all those hacks!
        da_rgbd = predictions[1]
        da_rgb = predictions[2]

        # soft label hack
        rgbd_target = torch.from_numpy(np.random.uniform(0, 0.3, (da_rgbd.size(0), 1)).astype(np.float32)).cuda()
        rgb_target = torch.from_numpy(np.random.uniform(0.7, 1.2, (da_rgbd.size(0), 1)).astype(np.float32)).cuda()

        # flip hack
        #if random.uniform(0, 1) < 0.3:
        #    da_rgbd = F.binary_cross_entropy(da_rgbd, rgb_target)
        #    da_rgb = F.binary_cross_entropy(da_rgb, rgbd_target)
        #else:
        da_rgbd = F.binary_cross_entropy(da_rgbd, rgbd_target)
        da_rgb = F.binary_cross_entropy(da_rgb, rgb_target)

        return task_loss + da_rgbd + da_rgb
