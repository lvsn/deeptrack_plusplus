import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


class DeepTrackConvNet(NetworkBase):
    def __init__(self, image_size=150):
        super(DeepTrackConvNet, self).__init__()

        filter_size_1 = 24
        embedding_size = 50
        half_filter_size_1 = int(filter_size_1/2)

        self.convA = nn.Conv2d(4, filter_size_1, 5)
        self.batchA = nn.BatchNorm2d(filter_size_1)

        self.convB = nn.Conv2d(4, filter_size_1, 5)
        self.batchB = nn.BatchNorm2d(filter_size_1)

        self.conv1 = nn.Conv2d(filter_size_1 * 2, filter_size_1 * 2, 3)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 2)
        self.conv2 = nn.Conv2d(filter_size_1 * 2, filter_size_1 * 2, 3)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 2)
        self.conv3 = nn.Conv2d(filter_size_1 * 2, filter_size_1 * 2, 3)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 2)
        
        self.view_size = 48 * 7 * 7
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout()

        self.criterion = nn.MSELoss()

    def forward(self, A, B):
        A = self.batchA(F.max_pool2d(F.elu(self.convA(A)), 2))

        B = self.batchB(F.max_pool2d(F.elu(self.convB(B)), 2))
        
        AB = torch.cat((A, B), 1)
        AB = self.batch1(F.max_pool2d(F.elu(self.conv1(AB)), 2))
        AB = self.batch2(F.max_pool2d(F.elu(self.conv2(AB)), 2))
        AB = self.batch3(F.max_pool2d(F.elu(self.conv3(AB)), 2))
        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = F.elu(self.fc1(AB))
        AB = F.tanh(self.fc2(AB))
        return AB

    def loss(self, predictions, targets):
        return self.criterion(predictions[0], targets[0])

if __name__ == '__main__':
    from pytorch_toolbox.probe.runtime import compute_single_test_time
    import matplotlib.pyplot as plt

    imsize = 150
    compute_single_test_time(DeepTrackConvNet, [(4, imsize, imsize), (4, imsize, imsize)],
                             batch_size=1, iterations=1000, is_cuda=True,
                             input_size=imsize)
    plt.show()
