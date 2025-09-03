import torch
import torch.nn as nn
import torchgeometry as tgm
import numpy as np

from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2


class PMLoss(nn.Module):
    """
    PyTorch Module for the Point Matching loss function

    Point Matching Loss introduced in the paper: 
    DeepIM: Deep Iterative Matching for 6D Pose Estimation (Li et al. 2018)
    https://github.com/NVlabs/DeepIM-PyTorch

    With inspiration from the implementation in:
    Self6D: Self-Supervised Monocular 6D Object Pose Estimation (Wang et al. 2020)
    https://github.com/THU-DA-6D-Pose-Group/self6dpp
    """
    def __init__(self, model_renderer: ModelRenderer2, translation_range=1, rotation_range=1, n_sample=3000, loss_func='L1'):
        super(PMLoss, self).__init__()
        self.model_renderer = model_renderer
        self.n_sample = n_sample
        if loss_func == 'L1':
            self.loss_func = nn.L1Loss()
        model_renderer.pre_sample_points(40000, preload=True)
        
        translation_range = float(translation_range)
        rotation_range = float(rotation_range)
        ranges = np.array([translation_range] * 3 + [rotation_range] * 3)
        self.ranges = torch.from_numpy(ranges).cuda().float()

        self.weight = (1000 / self.model_renderer.object_max_width) * 9220
        

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, 6) Tensor
            target: (B, 6) Tensor
        B is the batch size
        6 for the 6D pose (3D translation + 3D rotation)
            
        Returns:
            loss: (1)
        """
        input = input * self.ranges
        target = target * self.ranges

        batch_size = input.shape[0]
        #samples = self.model_renderer.get_random_pts_on_surface_batch(self.n_sample, batch_size)
        #samples = torch.from_numpy(samples).cuda().float()
        samples = self.model_renderer.get_pre_sampled_points(self.n_sample, batch_size)

        translations_src = input[:,:3].unsqueeze(-1)
        rotations_src = tgm.angle_axis_to_rotation_matrix(input[:,3:])
        rotations_src = rotations_src[:,:3,:3]
        new_pts_src = Transform.transform_pts_batch(samples, rotations_src, translations_src)

        translations_tgt = target[:,:3].unsqueeze(-1)
        rotations_tgt = tgm.angle_axis_to_rotation_matrix(target[:,3:])
        rotations_tgt = rotations_tgt[:,:3,:3]
        new_pts_tgt = Transform.transform_pts_batch(samples, rotations_tgt, translations_tgt)

        # src_cpy = new_pts_src.clone().detach().cpu().numpy()
        # tgt_cpy = new_pts_tgt.clone().detach().cpu().numpy()
        
        # import matplotlib.pyplot as plt
        # for i in range(batch_size):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(121, projection='3d')
        #     ax.scatter(src_cpy[i,:,0], src_cpy[i,:,1], src_cpy[i,:,2], c='b', s=0.5, marker='o')
        #     ax = fig.add_subplot(122, projection='3d')
        #     ax.scatter(tgt_cpy[i,:,0], tgt_cpy[i,:,1], tgt_cpy[i,:,2], c='r', s=0.5, marker='o')
        #     plt.show()
        
        loss = self.loss_func(torch.mul(new_pts_src, self.weight), torch.mul(new_pts_tgt, self.weight)) / self.n_sample
        return loss
