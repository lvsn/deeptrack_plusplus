import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


class DeepTrackZNet(NetworkBase):
    def __init__(self):
        super(DeepTrackZNet, self).__init__()

        filter_size_1 = 96
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1/2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.BatchNorm2d(filter_size_1)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.fireB = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2 = nn.BatchNorm2d(filter_size_1)

        self.fire1 = Fire(filter_size_1 * 2, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 4)
        self.fire2 = Fire(filter_size_1 * 4, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire3 = Fire(filter_size_1 * 8, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 8)

        self.view_size = filter_size_1 * 8 * 4 * 4
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout()

        self.criterion = nn.MSELoss()

    def forward(self, A, B):
        A = F.elu(self.batchA(self.convA(A)))
        A = self.batchA2(F.max_pool2d(self.fireA(A), 2))

        B = F.elu(self.batchB(self.convB(B)))
        B = self.batchB2(F.max_pool2d(self.fireB(B), 2))

        AB = torch.cat((A, B), 1)
        AB = self.batch1(F.max_pool2d(self.fire1(AB), 2))
        AB = self.batch2(F.max_pool2d(self.fire2(AB), 2))
        AB = self.batch3(F.max_pool2d(self.fire3(AB), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = F.elu(self.fc1(AB))
        AB = F.tanh(self.fc2(AB))
        return AB

    def loss(self, predictions, targets):
        diff = targets[1] * (predictions[0] - targets[0]) ** 2
        loss = torch.sum(diff, 1)/6
        loss = torch.mean(loss)
        return loss
