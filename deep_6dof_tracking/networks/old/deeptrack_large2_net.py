import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


def hook(grad):
    print(grad)


class DeepTrackLarge2Net(NetworkBase):
    def __init__(self):
        super(DeepTrackLarge2Net, self).__init__()

        filter_size_1 = 48
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(4, filter_size_1, 3)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)

        self.convB = nn.Conv2d(4, filter_size_1, 3)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.fireB = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)

        self.batch0 = nn.BatchNorm2d(filter_size_1 * 4)
        self.fire1 = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 4)
        self.fire2 = Fire(filter_size_1 * 4, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire3 = Fire(filter_size_1 * 8, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire4 = Fire(filter_size_1 * 8, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch4 = nn.BatchNorm2d(filter_size_1 * 8)

        self.view_size = filter_size_1 * 8 * 4 * 4
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout()
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)
        self.dropout_AB2 = nn.Dropout2d(0.3)

    def forward(self, A, B):
        A1 = F.elu(self.batchA(self.convA(A)))
        # self.probe_activation["A1"] = A
        A2 = self.fireA(A1)
        # self.probe_activation["A2"] = A

        B1 = F.elu(self.batchB(self.convB(B)))
        # self.probe_activation["B1"] = B
        B2 = self.fireB(B1)
        # self.probe_activation["B2"] = B

        AB = torch.cat((A1, B1, A2, B2), 1)
        AB = self.batch0(F.max_pool2d(AB, 2))
        AB = self.dropout_AB0(self.batch1(F.max_pool2d(self.fire1(AB), 2)))
        # self.probe_activation["AB1"] = AB
        AB = self.dropout_AB1(self.batch2(F.max_pool2d(self.fire2(AB), 2)))
        # self.probe_activation["AB2"] = AB
        AB = self.dropout_AB2(self.batch3(F.max_pool2d(self.fire3(AB), 2)))
        # self.probe_activation["AB3"] = AB
        AB = self.batch4(F.max_pool2d(self.fire4(AB), 2))

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

    compute_single_test_time(DeepTrackLarge2Net, [(4, 150, 150), (4, 150, 150)], 1000, is_cuda=True)
    plt.show()