import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


def hook(grad):
    print(grad)


class DeepTrackResNetMask(NetworkBase):
    def __init__(self, image_size=150, phase=None):
        super(DeepTrackResNetMask, self).__init__()

        filter_size_1 = 64
        self.filter_size_1 = filter_size_1
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.BatchNorm2d(filter_size_1 * 2)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.fireB = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2 = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1 = Fire(filter_size_1 * 4, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire2 = Fire(filter_size_1 * 8, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 16)
        self.fire3 = Fire(filter_size_1 * 16, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 24)

        view_width = int(int(int(int(int((image_size - 2)/2)/2)/2)/2)/2)

        self.view_size = filter_size_1 * 24 * view_width * view_width
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)

        # deconv
        self.de_fc1 = nn.Linear(embedding_size, self.view_size)
        self.de_conv1 = nn.Conv2d(3072, 1024, 3, padding=1)
        self.de_conv1_bn = nn.BatchNorm2d(1024)
        self.de_conv2 = nn.Conv2d(1024 + 1024, 512, 3, padding=1)
        self.de_conv2_bn = nn.BatchNorm2d(512)
        self.de_conv3 = nn.Conv2d(512 + 512, 256, 3, padding=1)
        self.de_conv3_bn = nn.BatchNorm2d(256)
        self.de_conv4 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.de_conv4_bn = nn.BatchNorm2d(128)
        self.de_conv5 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.de_conv5_bn = nn.BatchNorm2d(64)
        self.de_conv6 = nn.Conv2d(filter_size_1, 1, 3, padding=1)

    def forward(self, A, B):
        A_1 = self.batchA(F.elu(self.convA(A), inplace=True))
        A_2 = self.batchA2(F.max_pool2d(torch.cat((self.fireA(A_1), A_1), 1), 2))

        B_1 = self.batchB(F.elu(self.convB(B), inplace=True))
        B_2 = self.batchB2(F.max_pool2d(torch.cat((self.fireB(B_1), B_1), 1), 2))

        AB_0 = torch.cat((A_2, B_2), 1)
        AB_1 = self.dropout_AB0(self.batch1(F.max_pool2d(torch.cat((self.fire1(AB_0), AB_0), 1), 2)))
        AB_2 = self.dropout_AB1(self.batch2(F.max_pool2d(torch.cat((self.fire2(AB_1), AB_1), 1), 2)))
        AB_3 = self.batch3(F.max_pool2d(torch.cat((self.fire3(AB_2), AB_2), 1), 2))
        embedding = AB_3.view(-1, self.view_size)
        embedding = self.dropout1(embedding)
        embedding = self.fc_bn1(F.elu(self.fc1(embedding)))
        dof = torch.tanh(self.fc2(embedding))

        y = self.de_fc1(embedding)
        y = y.view(-1, AB_3.size(1), AB_3.size(2), AB_3.size(3))
        y = torch.cat((y, AB_3), 1)
        y = self.de_conv1_bn(F.elu(self.de_conv1(y)))
        y = F.upsample(y, AB_2.size()[2:], mode="bilinear")
        y = torch.cat((y, AB_2), 1)
        y = self.de_conv2_bn(F.elu(self.de_conv2(y)))
        y = F.upsample(y, AB_1.size()[2:], mode="bilinear")
        y = torch.cat((y, AB_1), 1)
        y = self.de_conv3_bn(F.elu(self.de_conv3(y)))
        y = F.upsample(y, B_2.size()[2:], mode="bilinear")
        y = torch.cat((y, B_2), 1)
        y = self.de_conv4_bn(F.elu(self.de_conv4(y)))
        y = F.upsample(y, B_1.size()[2:], mode="bilinear")
        y = torch.cat((y, B_1), 1)
        y = self.de_conv5_bn(F.elu(self.de_conv5(y)))
        y = F.upsample(y, B.size()[2:], mode="bilinear")
        # the output is normalised between -1 and 1
        y = torch.sigmoid(self.de_conv6(y))
        return dof, y[:, 0, :, :]

    def stream(self, A, B, batch_norms):
        A = self.batchA(F.elu(self.convA(A), inplace=True))
        A = self.batchA2(F.max_pool2d(torch.cat((self.fireA(A), A), 1), 2))

        B = self.batchB(F.elu(self.convB(B), inplace=True))
        B = self.batchB2(F.max_pool2d(torch.cat((self.fireB(B), B), 1), 2))

        AB = torch.cat((A, B), 1)
        AB = self.dropout_AB0(self.batch1(F.max_pool2d(torch.cat((self.fire1(AB), AB), 1), 2)))
        AB = self.dropout_AB1(self.batch2(F.max_pool2d(torch.cat((self.fire2(AB), AB), 1), 2)))
        AB = self.batch3(F.max_pool2d(torch.cat((self.fire3(AB), AB), 1), 2))

        AB = AB.view(-1, self.view_size)
        AB = self.dropout1(AB)
        embed = self.fc_bn1(F.elu(self.fc1(AB)))
        AB = torch.tanh(self.fc2(embed))
        return AB, embed

    def loss(self, predictions, targets):
        return nn.MSELoss()(predictions[0], targets[0]) + F.mse_loss(predictions[1], targets[1])
