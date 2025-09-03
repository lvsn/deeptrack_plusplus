import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire
from deep_6dof_tracking.networks.moddrop import ModDrop


def hook(grad):
    print(grad)


class DeepTrackResNetModDropGroupNorm(NetworkBase):
    def __init__(self, image_size=150, phase=None):
        super(DeepTrackResNetModDropGroupNorm, self).__init__()

        filter_size_1 = 64
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA = nn.GroupNorm(16, filter_size_1)
        self.fireA = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.GroupNorm(16, filter_size_1 * 2)

        self.moddropB = ModDrop([[0, 1, 2], [3]], dropout_proba=0.66)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchB = nn.GroupNorm(16, filter_size_1)
        self.fireB = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2 = nn.GroupNorm(16, filter_size_1 * 2)

        self.fire1 = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1 = nn.GroupNorm(16, filter_size_1 * 8)
        self.fire2 = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2 = nn.GroupNorm(16, filter_size_1 * 16)
        self.fire3 = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3 = nn.GroupNorm(16, filter_size_1 * 24)

        view_width = int(int(int(int(int((image_size - 2)/2)/2)/2)/2)/2)

        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

    def forward(self, A, B):
        AB = self.stream(A, B, None)
        return AB

    def stream(self, A, B, batch_norms):
        A = self.batchA(F.elu(self.convA(A), inplace=True))
        A = self.batchA2(F.max_pool2d(torch.cat((self.fireA(A), A), 1), 2))

        B = self.moddropB(B)
        B = self.batchB(F.elu(self.convB(B), inplace=True))
        B = self.batchB2(F.max_pool2d(torch.cat((self.fireB(B), B), 1), 2))

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0(self.batch1(F.max_pool2d(torch.cat((self.fire1(AB), AB), 1), 2)))
        AB = self.dropout_AB1(self.batch2(F.max_pool2d(torch.cat((self.fire2(AB), AB), 1), 2)))
        AB = self.batch3(F.max_pool2d(torch.cat((self.fire3(AB), AB), 1), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = self.fc_bn1(F.elu(self.fc1(AB)))
        latent = AB
        AB = torch.tanh(self.fc2(AB))
        return AB, latent

    def loss(self, predictions, targets):
        return nn.MSELoss()(predictions[0], targets[0])
