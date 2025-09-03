"""
Check angle magnitude distribution of dataset
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.utils.angles import euler2angle_axis

if __name__ == '__main__':
    dataset_path = "/media/ssd/deeptracking/dragon_norm_4/valid_synth"

    train_dataset = DeepTrackLoader(dataset_path)

    data_T = np.zeros((len(train_dataset.data_pair), 4))
    data_R = np.zeros((len(train_dataset.data_pair), 4))

    data_euler_T = np.zeros((len(train_dataset.data_pair), 3))
    data_euler_R = np.zeros((len(train_dataset.data_pair), 3))

    for i, (id, data) in tqdm(enumerate(train_dataset.data_pair.items())):
        frame, pose = data[0]
        pose = pose.to_parameters()
        magnitude_R, axis_R = euler2angle_axis(pose[5], pose[4], pose[3])
        magnitude_T, axis_T = euler2angle_axis(pose[2], pose[1], pose[0])
        data_T[i, 0] = magnitude_T
        data_T[i, 1:] = axis_T

        data_R[i, 0] = magnitude_R
        data_R[i, 1:] = axis_R

        data_euler_R[i, :] = pose[3:]
        data_euler_T[i, :] = pose[:3]

    for i in range(4):
        plt.hist(data_T[:, i])
        plt.show()
        plt.hist(data_R[:, i])
        plt.show()

    for i in range(3):
        plt.hist(data_euler_R[:, i])
        plt.show()
        plt.hist(data_euler_T[:, i])
        plt.show()


