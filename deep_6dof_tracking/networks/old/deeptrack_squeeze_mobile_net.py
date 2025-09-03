import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase


class FireMobile(nn.Module):
    """
    From SqueezeNet : https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

    """
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, stride=1):
        super(FireMobile, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1, stride=stride)
        self.expand1x1_activation = nn.ELU(inplace=True)
        self.expand3x3 = self.depth_wise_conv2d(squeeze_planes, expand3x3_planes, stride)
        self.expand3x3_activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

    @staticmethod
    def depth_wise_conv2d(input_channel, output_channel, stride):
        return nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, stride, 1, groups=input_channel),
            nn.BatchNorm2d(input_channel),
            #nn.ELU(inplace=True),

            nn.Conv2d(input_channel, output_channel, 1, 1, 0),
        )


class DeepTrackSqueezeMobileNet(NetworkBase):
    def __init__(self):
        super(DeepTrackSqueezeMobileNet, self).__init__()

        filter_size_1 = 96
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = FireMobile(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.BatchNorm2d(filter_size_1)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.fireB = FireMobile(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2 = nn.BatchNorm2d(filter_size_1)

        self.fire1 = FireMobile(filter_size_1 * 2, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 4)
        self.fire2 = FireMobile(filter_size_1 * 4, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire3 = FireMobile(filter_size_1 * 8, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 8)

        self.view_size = filter_size_1 * 8 * 4 * 4
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout()
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

    def forward(self, A, B):
        A = F.elu(self.batchA(self.convA(A)))
        # self.probe_activation["A1"] = A
        A = self.batchA2(F.max_pool2d(self.fireA(A), 2))
        # self.probe_activation["A2"] = A

        B = F.elu(self.batchB(self.convB(B)))
        # self.probe_activation["B1"] = B
        B = self.batchB2(F.max_pool2d(self.fireB(B), 2))
        # self.probe_activation["B2"] = B

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0(self.batch1(F.max_pool2d(self.fire1(AB), 2)))
        # self.probe_activation["AB1"] = AB
        AB = self.dropout_AB1(self.batch2(F.max_pool2d(self.fire2(AB), 2)))
        # self.probe_activation["AB2"] = AB
        AB = self.batch3(F.max_pool2d(self.fire3(AB), 2))
        # self.probe_activation["AB3"] = AB
        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = self.fc_bn1(F.elu(self.fc1(AB)))
        AB = F.tanh(self.fc2(AB))
        return AB

    def loss(self, predictions, targets):
        return nn.MSELoss()(predictions[0], targets[0])

if __name__ == '__main__':
    from pytorch_toolbox.probe.runtime import compute_single_test_time
    import matplotlib.pyplot as plt

    compute_single_test_time(DeepTrackSqueezeMobileNet, [(4, 150, 150), (4, 150, 150)], batch_size=10, iterations=700, is_cuda=True)
    plt.show()
