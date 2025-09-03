import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


def hook(grad):
    print(grad)


class DeepTrackResNet2Stream(NetworkBase):
    def __init__(self, image_size=150):
        super(DeepTrackResNet2Stream, self).__init__()

        filter_size_1 = 64
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        # RGB stream
        self.convA_rgb = nn.Conv2d(3, filter_size_1, 3, 2)
        self.batchA_rgb = nn.BatchNorm2d(filter_size_1)
        self.fireA_rgb = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1, skip_last_activation=True)
        self.batchA2_rgb = nn.BatchNorm2d(filter_size_1 * 2)

        self.convB_rgb = nn.Conv2d(3, filter_size_1, 3, 2)
        self.batchB_rgb = nn.BatchNorm2d(filter_size_1)
        self.fireB_rgb = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1, skip_last_activation=True)
        self.batchB2_rgb = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1_rgb = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2, skip_last_activation=True)
        self.batch1_rgb = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire2_rgb = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4, skip_last_activation=True)
        self.batch2_rgb = nn.BatchNorm2d(filter_size_1 * 16)
        self.fire3_rgb = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4, skip_last_activation=True)
        self.batch3_rgb = nn.BatchNorm2d(filter_size_1 * 24)

        # Depth stream
        self.convA_depth = nn.Conv2d(1, filter_size_1, 3, 2)
        self.batchA_depth = nn.BatchNorm2d(filter_size_1)
        self.fireA_depth = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2_depth = nn.BatchNorm2d(filter_size_1 * 2)

        self.convB_depth = nn.Conv2d(1, filter_size_1, 3, 2)
        self.batchB_depth = nn.BatchNorm2d(filter_size_1)
        self.fireB_depth = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2_depth = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1_depth = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1_depth = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire2_depth = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2_depth = nn.BatchNorm2d(filter_size_1 * 16)
        self.fire3_depth = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3_depth = nn.BatchNorm2d(filter_size_1 * 24)

        view_width = int(int(int(int(int((image_size - 2)/2)/2)/2)/2)/2)
        # FC part
        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

    def forward(self, A, B):

        A_rgb = A[:, :3, :, :]
        A_depth = A[:, 3, :, :].unsqueeze(1)

        B_rgb = B[:, :3, :, :]
        B_depth = B[:, 3, :, :].unsqueeze(1)

        A_depth = F.elu(self.convA_depth(A_depth), inplace=True)
        A_rgb = self.convA_rgb(A_rgb)
        A_rgb = self.batchA_rgb(A_depth * A_rgb + F.elu(A_rgb))  # Stream Link
        A_depth = self.batchB_depth(A_depth)

        A_depth = torch.cat((self.fireA_depth(A_depth), A_depth), 1)
        A_rgb = torch.cat((self.fireA_rgb(A_rgb), A_rgb), 1)
        A_rgb = A_depth * A_rgb + F.elu(A_rgb) # Stream Link
        A_depth = self.batchA2_depth(F.max_pool2d(A_depth, 2))
        A_rgb = self.batchA2_rgb(F.max_pool2d(A_rgb, 2))

        B_depth = F.elu(self.convB_depth(B_depth), inplace=True)
        B_rgb = self.convB_rgb(B_rgb)
        B_rgb = self.batchB_rgb(B_depth * B_rgb + F.elu(B_rgb))  # Stream Link
        B_depth = self.batchB_depth(B_depth)

        B_depth = torch.cat((self.fireB_depth(B_depth), B_depth), 1)
        B_rgb = torch.cat((self.fireB_rgb(B_rgb), B_rgb), 1)
        B_rgb = B_depth * B_rgb + F.elu(B_rgb)  # Stream Link
        B_depth = self.batchB2_depth(F.max_pool2d(B_depth, 2))
        B_rgb = self.batchB2_rgb(F.max_pool2d(B_rgb, 2))

        AB_rgb = torch.cat((A_rgb, B_rgb), 1)
        AB_depth = torch.cat((A_depth, B_depth), 1)

        AB_depth = torch.cat((self.fire1_depth(AB_depth), AB_depth), 1)
        AB_rgb = torch.cat((self.fire1_rgb(AB_rgb), AB_rgb), 1)
        AB_rgb = AB_depth * AB_rgb + F.elu(AB_rgb)  # Stream Link
        AB_depth = self.batch1_depth(F.max_pool2d(AB_depth, 2))
        AB_rgb = self.batch1_rgb(F.max_pool2d(AB_rgb, 2))

        AB_depth = torch.cat((self.fire2_depth(AB_depth), AB_depth), 1)
        AB_rgb = torch.cat((self.fire2_rgb(AB_rgb), AB_rgb), 1)
        AB_rgb = AB_depth * AB_rgb + F.elu(AB_rgb)  # Stream Link
        AB_depth = self.batch2_depth(F.max_pool2d(AB_depth, 2))
        AB_rgb = self.batch2_rgb(F.max_pool2d(AB_rgb, 2))

        AB_depth = torch.cat((self.fire3_depth(AB_depth), AB_depth), 1)
        AB_rgb = torch.cat((self.fire3_rgb(AB_rgb), AB_rgb), 1)
        AB_rgb = AB_depth * AB_rgb + F.elu(AB_rgb)  # Stream Link
        AB_rgb = self.batch3_rgb(F.max_pool2d(AB_rgb, 2))

        AB = AB_rgb.view(-1, self.view_size)
        AB = self.dropout1(AB)
        AB = self.fc_bn1(F.elu(self.fc1(AB)))
        AB = F.tanh(self.fc2(AB))
        return AB

    def loss(self, predictions, targets):
        return nn.MSELoss()(predictions[0], targets[0])

if __name__ == '__main__':
    from pytorch_toolbox.probe.runtime import compute_single_test_time
    import matplotlib.pyplot as plt

    imsize = 150
    compute_single_test_time(DeepTrackResNet, [(4, imsize, imsize), (4, imsize, imsize)],
                             batch_size=1, iterations=1000, is_cuda=True,
                             image_size=imsize)
    plt.show()
