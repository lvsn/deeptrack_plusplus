import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from deep_6dof_tracking.networks.fire_module import Fire

# REMOVEEE ALL
# import numpy as np

def hook(grad):
    print(grad)


class DeepTrackResNet(NetworkBase):
    def __init__(self, image_size=150, phase=None, fx=1.0, fy=1.0, delta_pose=False, loss_func=nn.MSELoss()):
        super(DeepTrackResNet, self).__init__()

        # REMOVEEE ALL
        # self.i = 0

        filter_size_1 = 64
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

        self.loss_func = loss_func

        self.fx = fx
        self.fy = fy
        self.delta_pose = delta_pose

    def forward(self, A, B, T_src=None):
        AB = self.stream(A, B, None, T_src=T_src)
        return AB

    def stream(self, A, B, batch_norms, T_src=None):
        # REMOVEEE ALL
        # AB_debug = torch.cat((A, B), 0)
        # torch.save(A, f'/home-local2/chren50.extra.nobkp/network_input_dog/A{self.i}.pt') 
        # torch.save(B, f'/home-local2/chren50.extra.nobkp/network_input_dog/B{self.i}.pt') 
        # np.save(f'/home-local2/chren50.extra.nobkp/network_input_clock/A{self.i}.npy', A.cpu().numpy())
        # np.save(f'/home-local2/chren50.extra.nobkp/network_input_clock/B{self.i}.npy', B.cpu().numpy())
        # self.i += 1

        A = self.batchA(F.elu(self.convA(A), inplace=True))
        A = self.batchA2(F.max_pool2d(torch.cat((self.fireA(A), A), 1), 2))

        B = self.batchB(F.elu(self.convB(B), inplace=True))
        #self.probe_activation["moddrop"] = B
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

        if self.delta_pose:
            AB = self.delta_transform(T_src, AB)
        return AB, latent

    def loss(self, predictions, targets):
        loss = self.loss_func(predictions[0], targets[0])
        return loss
    
    def delta_transform(self, T_src, T_delta):
        '''
        :param T_src: (x1, y1, z1)
        :param T_delta: (dx, dy, dz)
        :return: T_tgt: (x2, y2, z2)
        '''
        # #weight = 10
        # weight = 1
        # vz = torch.div(T_src[:, 2], torch.exp(T_delta[:, 2] / weight))
        # vx = torch.mul(vz, (T_delta[:, 0] / self.fx) + (T_src[:, 0]/T_src[:, 2]))
        # vy = torch.mul(vz, (T_delta[:, 1] / self.fy) + (T_src[:, 1]/T_src[:, 2]))
        # #vx = torch.mul(vz, torch.addcdiv(T_delta[:, 0::3] / weight, 1.0, T_src[:, 0::3], T_src[:, 2::3]))
        # # vy = torch.mul(vz, torch.addcdiv(T_delta[:, 1::3] / weight, 1.0, T_src[:, 1::3], T_src[:, 2::3]))

        # OK CHECK ICI
        weight = 10
        # weight = 1
        vz = torch.div(T_src[:, 2], torch.exp(T_delta[:, 2] / weight))
        vx = torch.mul(vz, (T_delta[:, 0] / weight) + (T_src[:, 0]/T_src[:, 2]))
        vy = torch.mul(vz, (T_delta[:, 1] / weight) + (T_src[:, 1]/T_src[:, 2]))
        #vx = torch.mul(vz, torch.addcdiv(T_delta[:, 0::3] / weight, 1.0, T_src[:, 0::3], T_src[:, 2::3]))
        # vy = torch.mul(vz, torch.addcdiv(T_delta[:, 1::3] / weight, 1.0, T_src[:, 1::3], T_src[:, 2::3]))


        T_delta_new = torch.zeros_like(T_delta)
        T_delta_new[:, 0] = vx
        T_delta_new[:, 1] = vy
        T_delta_new[:, 2] = vz
        T_delta_new[:, 3:6] = T_delta[:, 3:6]

        return T_delta_new
