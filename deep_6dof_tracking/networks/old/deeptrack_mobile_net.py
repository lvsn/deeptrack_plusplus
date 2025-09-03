import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase


class DeepTrackMobileNet(NetworkBase):
    def __init__(self):
        super(DeepTrackMobileNet, self).__init__()

        filter_size_1 = 32
        embedding_size = 500

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.convA1 = self.depth_wise_conv2d(filter_size_1, filter_size_1*2, 2)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.convB1 = self.depth_wise_conv2d(filter_size_1, filter_size_1*2, 2)

        self.convAB = self.depth_wise_conv2d(filter_size_1 * 4, filter_size_1 * 4, 2)
        self.convAB1 = self.depth_wise_conv2d(filter_size_1 * 4, filter_size_1 * 4, 2)
        self.convAB2 = self.depth_wise_conv2d(filter_size_1 * 4, filter_size_1 * 8, 2)

        self.view_size = filter_size_1 * 8 * 5 * 5
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout()
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

    @staticmethod
    def depth_wise_conv2d(input_channel, output_channel, stride):
        return nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, stride, 1, groups=input_channel),
            nn.BatchNorm2d(input_channel),
            nn.ELU(inplace=True),

            nn.Conv2d(input_channel, output_channel, 1, 1, 0),
            nn.BatchNorm2d(output_channel),
            nn.ELU(inplace=True),
        )

    def forward(self, A, B):
        A = F.elu(self.batchA(self.convA(A)))
        A = self.convA1(A)

        B = F.elu(self.batchB(self.convB(B)))
        B = self.convB1(B)

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0(self.convAB(AB))
        AB = self.dropout_AB1(self.convAB1(AB))
        AB = self.convAB2(AB)
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

    compute_single_test_time(DeepTrackMobileNet, [(4, 150, 150), (4, 150, 150)], batch_size=10, iterations=700, is_cuda=True)
    plt.show()