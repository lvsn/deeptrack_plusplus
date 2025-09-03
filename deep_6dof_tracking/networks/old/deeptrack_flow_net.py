import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from pytorch_toolbox.network_base import NetworkBase

from deep_6dof_tracking.networks.fire_module import Fire
from deep_6dof_tracking.networks.rodrigues_function import RodriguesFunction

DEBUG = False


def coord_to_image(coords):
    height = 480
    width = 640
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pixels = coords.data.cpu().numpy().astype(int).T
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    img[pixels[:, 1], pixels[:, 0], 0] = 255
    return img


def hook(grad):
    print(grad)


class DeepTrackFlowNet(NetworkBase):
    def __init__(self, show_mask=False):
        super(DeepTrackFlowNet, self).__init__()

        filter_size_1 = 96
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.BatchNorm2d(filter_size_1)

        self.convB = nn.Conv2d(4, filter_size_1, 3, 2)
        self.batchB = nn.BatchNorm2d(filter_size_1)
        self.fireB = Fire(filter_size_1, half_filter_size_1, half_filter_size_1, half_filter_size_1)
        self.batchB2 = nn.BatchNorm2d(filter_size_1)

        self.fire1 = Fire(filter_size_1 * 2, filter_size_1, filter_size_1 * 2, filter_size_1 * 2)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 4)
        self.fire2 = Fire(filter_size_1 * 4, filter_size_1 * 2, filter_size_1 * 4, filter_size_1 * 4)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire3 = Fire(filter_size_1 * 8, filter_size_1 * 4, filter_size_1 * 4, filter_size_1 * 4)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 8)

        self.view_size = filter_size_1 * 8 * 4 * 4
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        # deconv
        self.de_fc1 = nn.Linear(embedding_size, self.view_size)
        self.de_conv1 = nn.Conv2d(filter_size_1*8*2, filter_size_1*8, 3, padding=1)
        self.de_conv1_bn = nn.BatchNorm2d(filter_size_1*8)
        self.de_conv2 = nn.Conv2d(filter_size_1*8*2, filter_size_1*4, 3, padding=1)
        self.de_conv2_bn = nn.BatchNorm2d(filter_size_1*4)
        self.de_conv3 = nn.Conv2d(filter_size_1*4*2, filter_size_1*2, 3, padding=1)
        self.de_conv3_bn = nn.BatchNorm2d(filter_size_1*2)
        self.de_conv4 = nn.Conv2d(filter_size_1*3, filter_size_1, 3, padding=1)
        self.de_conv4_bn = nn.BatchNorm2d(filter_size_1)
        self.de_conv5 = nn.Conv2d(filter_size_1*2, filter_size_1, 3, padding=1)
        self.de_conv5_bn = nn.BatchNorm2d(filter_size_1)
        self.de_conv6 = nn.Conv2d(filter_size_1, 2, 3, padding=1)

        self.dropout1 = nn.Dropout()
        self.dropout_AB0 = nn.Dropout2d(0.3)
        self.dropout_AB1 = nn.Dropout2d(0.3)
        self.show_mask = show_mask

    def forward(self, A_0, B_0):
        A_1 = F.elu(self.batchA(self.convA(A_0)))
        A_2 = self.batchA2(F.max_pool2d(self.fireA(A_1), 2))

        B_1 = F.elu(self.batchB(self.convB(B_0)))
        B_2 = self.batchB2(F.max_pool2d(self.fireB(B_1), 2))

        AB_0 = torch.cat((A_2, B_2), 1)
        AB_1 = self.dropout_AB0(self.batch1(F.max_pool2d(self.fire1(AB_0), 2)))
        AB_2 = self.dropout_AB1(self.batch2(F.max_pool2d(self.fire2(AB_1), 2)))
        AB_3 = self.batch3(F.max_pool2d(self.fire3(AB_2), 2))

        embedding = AB_3.view(-1, self.view_size)
        embedding = self.dropout1(embedding)
        embedding = self.fc_bn1(F.elu(self.fc1(embedding)))

        delta_pose = self.fc2(embedding)
        # AB.register_hook(hook)
        delta_pose = F.tanh(delta_pose)
        # Deconvolution
        if self.training or self.show_mask:
            y = self.de_fc1(embedding)
            y = y.view(-1, 96 * 8, 4, 4)
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
            y = F.upsample(y, B_0.size()[2:], mode="bilinear")
            # the output is normalised between -1 and 1
            y = F.tanh(self.de_conv6(y))
            if self.show_mask:
                import cv2
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('seismic')

                mask_x = (y.data[0].cpu().numpy()[0, :, :] + 1) / 2
                mask_x = (cmap(mask_x.T)*255).astype(np.uint8)

                mask_y = (y.data[0].cpu().numpy()[1, :, :] + 1) / 2
                mask_y = (cmap(mask_y.T)*255).astype(np.uint8)
                cv2.imshow("mask flow net", np.concatenate((mask_x, mask_y), axis=1))
                return delta_pose
            return delta_pose, y

        return delta_pose

    def loss(self, predictions, targets):
        loss_pose = nn.MSELoss()(predictions[0], targets[0])
        loss_decoder = 0
        if len(predictions) == 2:
            loss_decoder = nn.MSELoss()(predictions[1], targets[1])
        return loss_pose + loss_decoder
