import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire


class DeepTrackCompositionNet(NetworkBase):
    def __init__(self):
        super(DeepTrackCompositionNet, self).__init__()
        filter_size_1 = 24
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1/2)
        self.activations = []

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

        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout()

    def forward(self, A, B, B_mask=None, mask0=None, mask1=None, mask2=None, mask3=None, mask4=None):
        o_B1, o_B2, o_AB1, o_AB2, o_AB3, o_ABL = self.deep_track(A, B)

        if B_mask is not None:
            m_B1, m_B2, m_AB1, m_AB2, m_AB3, m_ABL = self.deep_track(A, B_mask, mask0, mask1, mask2, mask3, mask4)
            return o_ABL, m_ABL, \
                   o_B1, o_B2, o_AB1, o_AB2, o_AB3, \
                   m_B1, m_B2, m_AB1, m_AB2, m_AB3
        else:
            return o_ABL

    def deep_track(self, A, B, mask0=None, mask1=None, mask2=None, mask3=None, mask4=None):
        """
                sizes :     B1 = 24x74x74
                            B2 = 24x37x37
                            AB1 = 48x18x18
                            AB2 = 48x9x9
                            AB3 = 48x4x4
                :param A:
                :param B:
                :return:
                """
        A1 = F.elu(self.batchA(self.convA(A)))
        A2 = self.batchA2(F.max_pool2d(self.fireA(A1), 2))

        B1 = F.elu(self.batchB(self.convB(B)))
        if mask0 is not None:
            B1_m = torch.mul(B1, mask0.unsqueeze(1))
            B2 = self.batchB2(F.max_pool2d(self.fireB(B1_m), 2))
        else:
            B2 = self.batchB2(F.max_pool2d(self.fireB(B1), 2))

        if mask1 is not None:
            B2_m = torch.mul(B2, mask1.unsqueeze(1))
            AB = torch.cat((A2, B2_m), 1)
        else:
            AB = torch.cat((A2, B2), 1)
        AB1 = self.batch1(F.max_pool2d(self.fire1(AB), 2))
        if mask2 is not None:
            AB1_m = torch.mul(AB1, mask2.unsqueeze(1))
            AB2 = self.batch2(F.max_pool2d(self.fire2(AB1_m), 2))
        else:
            AB2 = self.batch2(F.max_pool2d(self.fire2(AB1), 2))
        if mask3 is not None:
            AB2_m = torch.mul(AB2, mask3.unsqueeze(1))
            AB3 = self.batch3(F.max_pool2d(self.fire3(AB2_m), 2))
        else:
            AB3 = self.batch3(F.max_pool2d(self.fire3(AB2), 2))
        if mask4 is not None:
            AB3_m = torch.mul(AB3, mask4.unsqueeze(1))
            ABL = AB3_m.view(-1, self.view_size)
        else:
            ABL = AB3.view(-1, self.view_size)
        ABL = self.dropout1(ABL)
        ABL = self.fc_bn1(F.elu(self.fc1(ABL)))
        ABL = F.tanh(self.fc2(ABL))
        return B1, B2, AB1, AB2, AB3, ABL

    def loss(self, predictions, targets):

        activation_losses = []

        o_task_loss = nn.MSELoss()(predictions[0], targets[0])
        m_task_loss = nn.MSELoss()(predictions[1], targets[0])

        for i in range(5):
            prediction = torch.mul(predictions[2+i], targets[i+1].unsqueeze(1))  # mask the activation
            target = predictions[2 + i + 5]
            l = torch.sum((prediction - target) ** 2) / predictions[2+i].data.nelement()
            activation_losses.append(l)

        activation_loss = activation_losses[0] * 0
        activation_loss += activation_losses[1] * 0
        activation_loss += activation_losses[2] * 0
        activation_loss += activation_losses[3] * 0.5
        activation_loss += activation_losses[4] * 1
        loss = 0.5*o_task_loss + 0.5*m_task_loss + activation_loss
        return loss
