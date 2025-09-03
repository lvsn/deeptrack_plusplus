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
    print(torch.norm(grad[:, :3], 2, 1))
    print(torch.norm(grad[:, 3:], 2, 1))


class DeepTrackGeoNet(NetworkBase):
    def __init__(self, image_size=150):
        super(DeepTrackGeoNet, self).__init__()

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

        view_width = int(int(int(int(int((image_size - 2) / 2) / 2) / 2) / 2) / 2)

        self.view_size = filter_size_1 * 8 * view_width * view_width
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
        AB = self.fc2(AB)
        # AB.register_hook(hook)
        AB_T = F.tanh(AB[:, :3]) * 0.05
        AB_R = F.tanh(AB[:, 3:])
        return torch.cat((AB_T, AB_R), 1)

    def loss(self, predictions, targets):
        delta_pose, model, previous_pose, K, debugA, debugB, _, _ = targets
        predictions = predictions[0]

        # rescale the outputs
        """
        translation_range = translation_range.expand(translation_range.size(0), 3)
        rotation_range = rotation_range.expand(rotation_range.size(0), 3)
        gt_translation = torch.mul(delta_pose[:, :3], translation_range)
        gt_rotation = torch.mul(delta_pose[:, 3:], rotation_range)
        prediction_translation = torch.mul(predictions[:, :3], translation_range)
        prediction_rotation = torch.mul(predictions[:, 3:], rotation_range)
        """
        gt_translation = delta_pose[:, :3]
        gt_rotation = delta_pose[:, 3:]
        prediction_translation = predictions[:, :3]
        prediction_rotation = predictions[:, 3:]

        R = RodriguesFunction()(previous_pose[:, 3:]).view(-1, 3, 3)
        T = previous_pose[:, :3]

        gt_delta_R = RodriguesFunction()(gt_rotation).view(-1, 3, 3)
        gt_delta_T = gt_translation
        gt_R = torch.bmm(gt_delta_R, R)
        gt_T = T + gt_delta_T

        pred_delta_R = RodriguesFunction()(prediction_rotation).view(-1, 3, 3)
        pred_delta_T = prediction_translation

        # Apply prediction Rotation and Translation
        pred_R = torch.bmm(pred_delta_R, R)
        pred_T = T + pred_delta_T

        model_R_gt = gt_T.unsqueeze(-1) + torch.bmm(gt_R, model)
        model_T_gt = gt_T.unsqueeze(-1) + torch.bmm(gt_R, model)

        model_R = gt_T.unsqueeze(-1) + torch.bmm(pred_R, model)
        model_T = pred_T.unsqueeze(-1) + torch.bmm(gt_R, model)

        model_R = torch.bmm(K, model_R)
        model_T = torch.bmm(K, model_T)
        model_R_gt = torch.bmm(K, model_R_gt)
        model_T_gt = torch.bmm(K, model_T_gt)

        model_R_gt_XY = model_R_gt[:, :2, :] / model_R_gt[:, 2, :].unsqueeze(1)
        model_R_XY = model_R[:, :2, :] / model_R[:, 2, :].unsqueeze(1)
        model_T_gt_XY = model_T_gt[:, :2, :] / model_T_gt[:, 2, :].unsqueeze(1)
        model_T_XY = model_T[:, :2, :] / model_T[:, 2, :].unsqueeze(1)

        # scale to mm, much closer to pixel error
        model_R_gt_Z = model_R_gt[:, 2, :] * 1000
        model_R_Z = model_R[:, 2, :] * 1000
        model_T_gt_Z = model_T_gt[:, 2, :] * 1000
        model_T_Z = model_T[:, 2, :] * 1000

        if DEBUG:
            import matplotlib.pyplot as plt
            for i in range(1):
                R_prediction = coord_to_image(model_R_XY[i])
                R_gt = coord_to_image(model_R_gt_XY[i])
                R_prediction[R_gt[:, :, 0] == 255, :] = [0, 255, 0]
                T_prediction = coord_to_image(model_T_XY[i])
                T_gt = coord_to_image(model_T_gt_XY[i])
                T_prediction[T_gt[:, :, 0] == 255, :] = [0, 255, 0]
                plt.subplot("121")
                plt.imshow(np.concatenate((R_prediction, T_prediction), axis=1))
                plt.subplot("122")
                plt.imshow(np.concatenate((debugA[i].data.cpu().numpy(), debugB[i].data.cpu().numpy()), axis=1))
                plt.show()

        norm_n = 2
        # mse_R = torch.sum(torch.abs(projected_R_prediction_model[:, :2, :] - projected_R_gt_model[:, :2, :]), 1) / 2
        # mse_T = torch.sum(torch.abs(projected_T_prediction_model[:, :2, :] - projected_T_gt_model[:, :2, :]), 1) / 2
        # compute mean square error
        mse_R = ((model_R_XY[:, 0, :] - model_R_gt_XY[:, 0, :]) ** norm_n + \
                 (model_R_XY[:, 1, :] - model_R_gt_XY[:, 1, :]) ** norm_n + \
                 (model_R_Z[:, :] - model_R_gt_Z[:, :]) ** norm_n) / 3

        mse_T = ((model_T_XY[:, 0, :] - model_T_gt_XY[:, 0, :]) ** norm_n + (
            model_T_XY[:, 1, :] - model_T_gt_XY[:, 1, :]) ** norm_n + (model_T_Z[:, :] - model_T_gt_Z[:, :]) ** norm_n) / 3


        mean_mse_T = mse_T.mean()
        mean_mse_R = mse_R.mean()
        loss = mean_mse_T + mean_mse_R

        return loss
