import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
import numpy as np

from deep_6dof_tracking.networks.fire_module import Fire
DEBUG = False
if DEBUG:
    import math
    import matplotlib.pyplot as plt


class DimLogSoftMax(nn.Module):
    def forward(self, input_, is_log=True):
        func = F.log_softmax
        if not is_log:
            func = F.softmax
        dimension = input_.size()[1]
        output_ = torch.stack([func(input_[:, i, :]) for i in range(dimension)], 1)
        return output_


def show_gradient(grad):
    numpy_grad = grad.data.cpu().numpy()
    sum_grad = np.abs(np.sum(numpy_grad, axis=(0, 1)))
    mean_grad = np.mean(numpy_grad, axis=(0, 1))
    print(mean_grad)
    import matplotlib.pyplot as plt
    plt.bar(np.arange(len(mean_grad)), mean_grad)
    plt.show()


class DeepTrackBinGeoNet(NetworkBase):
    def __init__(self, n_bins, log_softmax=True, proba=False):
        super(DeepTrackBinGeoNet, self).__init__()
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
        self.activations = []

        self.translation_centers = None
        self.rotation_centers = None
        self.proba = proba

    def forward(self, A, B):
        A = F.elu(self.batchA(self.convA(A)))
        if DEBUG:
            #self.show_activations(A)
            self.activations.append(A.data.cpu().numpy())
        A = self.batchA2(F.max_pool2d(self.fireA(A), 2))
        if DEBUG:
            #self.show_activations(A)
            self.activations.append(A.data.cpu().numpy())

        B = F.elu(self.batchB(self.convB(B)))
        if DEBUG:
            #self.show_activations(B)
            self.activations.append(B.data.cpu().numpy())
        B = self.batchB2(F.max_pool2d(self.fireB(B), 2))
        if DEBUG:
            #self.show_activations(B)
            self.activations.append(B.data.cpu().numpy())

        AB = torch.cat((A, B), 1)
        AB = self.batch1(F.max_pool2d(self.fire1(AB), 2))
        if DEBUG:
            #self.show_activations(AB)
            self.activations.append(AB.data.cpu().numpy())
        AB = self.batch2(F.max_pool2d(self.fire2(AB), 2))
        if DEBUG:
            #self.show_activations(AB)
            self.activations.append(AB.data.cpu().numpy())
        AB = self.batch3(F.max_pool2d(self.fire3(AB), 2))
        if DEBUG:
            #self.show_activations(AB)
            self.activations.append(AB.data.cpu().numpy())

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = self.dropout2(self.fc_bn1(F.elu(self.fc1(AB))))
        AB = self.fc2(AB)
        AB = AB.view(-1, 6, self.n_bins)
        AB = self.log_softmax(AB, self.is_log_softmax)
        AB.register_hook(show_gradient)
        if self.proba:
            return AB[:, 0, :], AB[:, 1, :], AB[:, 2, :], AB[:, 3, :], AB[:, 4, :], AB[:, 5, :]
        translation = torch.sum(AB[:, :3, :] * self.translation_centers, 2)
        rotation = torch.sum(AB[:, 3:, :] * self.rotation_centers, 2)
        poses = torch.cat((translation, rotation), 1)
        return poses

    def show_activations(self, feature):
        root = int(math.sqrt(feature.size(1)))
        fig, axes = plt.subplots(root, root)
        for i in range(root):
            for j in range(root):
                ax = axes[i][j]
                ax.imshow(feature.data.cpu().numpy()[0, i * root + j, :, :])
        plt.show()

    def loss(self, predictions, targets):
        loss = nn.MSELoss()(predictions[0], targets[0])
        return loss

    def set_bins(self, translation_bins, rotation_bins, translation_range, rotation_range):

        t_centers = self.compute_bin_center(translation_bins, translation_range)
        r_centers = self.compute_bin_center(rotation_bins, rotation_range)
        self.translation_centers = Variable(torch.from_numpy(t_centers.astype(np.float32)))
        self.translation_centers = self.translation_centers.cuda()
        self.rotation_centers = Variable(torch.from_numpy(r_centers.astype(np.float32)))
        self.rotation_centers = self.rotation_centers.cuda()

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

