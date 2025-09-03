import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase

from deep_6dof_tracking.networks.fire_module import Fire
DEBUG = False


class DimLogSoftMax(nn.Module):
    def forward(self, input_, is_log=True):
        func = F.log_softmax
        if not is_log:
            func = F.softmax
        dimension = input_.size()[1]
        output_ = torch.stack([func(input_[:, i, :]) for i in range(dimension)], 1)
        return output_


def show_gradient(grad):
    print(torch.mean(torch.norm(grad, 2, 2)/torch.norm(grad, 2)))
    #print(torch.norm(grad, 2))


class DeepTrackBinNet(NetworkBase):
    def __init__(self, n_bins, log_softmax=True):
        super(DeepTrackBinNet, self).__init__()
        self.n_bins = n_bins
        filter_size_1 = 24
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

        self.fire1 = Fire(filter_size_1*2, filter_size_1, filter_size_1, filter_size_1)
        self.batch1 = nn.BatchNorm2d(filter_size_1*2)
        self.fire2 = Fire(filter_size_1*2, filter_size_1, filter_size_1, filter_size_1)
        self.batch2 = nn.BatchNorm2d(filter_size_1*2)
        self.fire3 = Fire(filter_size_1*2, filter_size_1, filter_size_1, filter_size_1)
        self.batch3 = nn.BatchNorm2d(filter_size_1*2)

        self.view_size = filter_size_1 * 2 * 4 * 4
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)

        self.fc2 = nn.Linear(embedding_size, self.n_bins*6)
        self.log_softmax = DimLogSoftMax()

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.is_log_softmax = log_softmax

    def forward(self, A, B):
        A = F.elu(self.batchA(self.convA(A)))
        self.probe_activation["A1"] = A
        A = self.batchA2(F.max_pool2d(self.fireA(A), 2))
        self.probe_activation["A2"] = A

        B = F.elu(self.batchB(self.convB(B)))
        self.probe_activation["B1"] = B
        B = self.batchB2(F.max_pool2d(self.fireB(B), 2))
        self.probe_activation["B2"] = B

        AB = torch.cat((A, B), 1)
        AB = self.batch1(F.max_pool2d(self.fire1(AB), 2))
        self.probe_activation["AB1"] = AB
        AB = self.batch2(F.max_pool2d(self.fire2(AB), 2))
        self.probe_activation["AB2"] = AB
        AB = self.batch3(F.max_pool2d(self.fire3(AB), 2))
        self.probe_activation["AB3"] = AB
        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = self.dropout2(self.fc_bn1(F.elu(self.fc1(AB))))
        AB = self.fc2(AB)
        AB = AB.view(-1, 6, self.n_bins)
        if DEBUG:
            AB.register_hook(show_gradient)
        AB = self.log_softmax(AB, self.is_log_softmax)
        return AB[:, 0, :], AB[:, 1, :], AB[:, 2, :], AB[:, 3, :], AB[:, 4, :], AB[:, 5, :]

    def loss(self, predictions, targets):
        l_tx = nn.KLDivLoss()(predictions[0], targets[0])
        l_ty = nn.KLDivLoss()(predictions[1], targets[1])
        l_tz = nn.KLDivLoss()(predictions[2], targets[2])

        l_rx = nn.KLDivLoss()(predictions[3], targets[3])
        l_ry = nn.KLDivLoss()(predictions[4], targets[4])
        l_rz = nn.KLDivLoss()(predictions[5], targets[5])



        total_loss = torch.cat([l_tx, l_ty, l_tz, l_rx, l_ry, l_rz])
        return torch.sum(total_loss)
