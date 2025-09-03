import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


class DeepTrackDenseNet(NetworkBase):
    def __init__(self, image_size=150):
        super(DeepTrackDenseNet, self).__init__()

        filter_size_1 = 24
        half_filter_size_1 = int(filter_size_1/2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, padding=1)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = Fire(filter_size_1 + 4, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.BatchNorm2d(filter_size_1*2 + 4)

        self.convB = nn.Conv2d(4, filter_size_1, 3, padding=1)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.fireB = Fire(filter_size_1 + 4, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2 = nn.BatchNorm2d(filter_size_1*2 + 4)

        self.fire1 = Fire(filter_size_1*4 + 8, filter_size_1, filter_size_1, filter_size_1)
        self.batch1 = nn.BatchNorm2d(filter_size_1*6 + 8)
        self.fire2 = Fire(filter_size_1*6 + 8, filter_size_1, filter_size_1, filter_size_1)
        self.batch2 = nn.BatchNorm2d(filter_size_1*8 + 8)
        self.fire3 = Fire(filter_size_1*8 + 8, filter_size_1, filter_size_1, filter_size_1)
        self.batch3 = nn.BatchNorm2d(filter_size_1*2)

        self.view_size = filter_size_1*2 * 10 * 10
        self.fc1 = nn.Linear(self.view_size, 50)
        self.fc_bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 6)

        self.dropout1 = nn.Dropout()
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

        self.criterion = nn.MSELoss()

    def forward(self, A0, B0):
        A1 = F.elu(self.batchA(self.convA(A0)), inplace=True)     # 4 -> 24
        A10 = torch.cat((A1, A0), 1)                # 28
        A2 = self.fireA(A10)                        # 28 -> 24
        A210 = torch.cat((A2, A10), 1)              # 52
        A210 = self.batchA2(F.max_pool2d(A210, 2))

        B1 = F.elu(self.batchB(self.convB(B0)), inplace=True)
        B10 = torch.cat((B1, B0), 1)
        B2 = self.fireB(B10)
        B210 = torch.cat((B2, B10), 1)
        B210 = self.batchB2(F.max_pool2d(B210, 2))

        AB0 = torch.cat((A210, B210), 1)            # 104
        AB1 = self.fire1(AB0)
        AB10 = torch.cat((AB1, AB0), 1)             # 104 + 24 -> 128
        AB10 = self.batch1(F.max_pool2d(AB10, 2))
        AB2 = self.fire2(AB10)
        AB210 = torch.cat((AB2, AB10), 1)
        AB210 = self.batch2(F.max_pool2d(AB210, 2))
        AB = self.batch3(F.max_pool2d(self.fire3(AB210), 2))
        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = F.elu(self.fc1(AB), inplace=True)
        AB = torch.tanh(self.fc2(AB))
        return AB

    def loss(self, predictions, targets):
        return self.criterion(predictions[0], targets[0])
