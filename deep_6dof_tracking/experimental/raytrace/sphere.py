import torch.nn as nn
import torch
from deeptracking.data.dataset_utils import combine_view_transform

from deep_6dof_tracking.networks.rodrigues_function import RodriguesFunction


class Sphere(nn.Module):
    """
        Apply Rigid 3D transform to pointcloud

    """
    def __init__(self, r):
        super(Sphere, self).__init__()
        self.r = r

    def forward(self, x, y, z):
        out = torch.sqrt(x.pow(2) + y.pow(2) + z.pow(2)) - self.r
        return out
