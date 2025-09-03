import argparse

import pandas as pd
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib
import os

from deep_6dof_tracking.utils.evaluation import get_pose_difference
from deep_6dof_tracking.utils.transform import Transform

matplotlib.style.use('ggplot')


def plot_2d(ground_truth, prediction, labels, name):
    fig, axes = plt.subplots(2, 3)
    X = np.arange(len(ground_truth))
    for i in range(6):
        col = int(i / 3)
        row = int(i % 3)
        axes[col, row].plot(X, ground_truth[:, i], color="blue", label='Ground Truth')
        axes[col, row].plot(X, prediction[:, i], color="red", label='Prediction')
        axes[col, row].set_title(labels[i])
    plt.legend()
    fig.savefig(os.path.join(figure_path, "Sequence_Pose_{}.png".format(name)))


def get_pose_differences(prediction_df, gt_df):
    diffs = []
    for pred, gt in zip(prediction_df.as_matrix(), gt_df.as_matrix()):
        pred_transform = Transform.from_parameters(*pred)
        gt_transform = Transform.from_parameters(*gt)
        diff = get_pose_difference(pred_transform, gt_transform)
        diffs.append(diff)
    return np.array(diffs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate test result')
    parser.add_argument('-p', '--path', help="information path", metavar="FILE")
    arguments = parser.parse_args()
    path = arguments.path

    prediction_file = os.path.join(path, "prediction_pose.csv")
    ground_truth_file = os.path.join(path, "ground_truth_pose.csv")
    projection_error_file = os.path.join(path, "projection_error.csv")

    max_rotation = 20
    max_translation = 0.02

    df_pred = pd.read_csv(prediction_file)
    df_gt = pd.read_csv(ground_truth_file)
    df_projection = pd.read_csv(projection_error_file)

    N = len(df_pred)
    diffs = get_pose_differences(df_pred, df_gt)

    rmse = np.sqrt(np.mean(np.square(diffs), axis=0))
    print(rmse[:3]*1000)
    print(rmse[3:])

    # Frequency analysis
    #signal = df.as_matrix(columns=[column_tags[2]])
    #yf = np.fft.fft(signal)/N
    #yf = np.fft.fftshift(yf)
    #xf = np.fft.fftfreq(N, 0.06)
    #axes[2].plot(xf, yf)
    #axes[2].set_ylim(0, 0.00003)

    # Diff analysis
    plt.figure(0)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    for i in range(3):
        axes[0].plot(np.arange(N), diffs[:, i])
        axes[1].plot(np.arange(N), diffs[:, i+3])
        axes[0].set_ylim(0, max_translation)
        axes[1].set_ylim(0, max_rotation)
    axes[2].plot(np.arange(N), df_projection.as_matrix())
    axes[2].set_ylim(0, 30)
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Mean pixel error")
    fig = plt.gcf()
    #plt.show()


    # Diff mean Translation/Rotation
    """
    plt.figure(1)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    #t_mean = df[['Tx', 'Ty', 'Tz']].mean(axis=1).as_matrix()
    t_mean = np.sqrt(df[['Tx']].as_matrix() ** 2 + df[['Ty']].as_matrix() ** 2 + df[['Tz']].as_matrix() ** 2)
    r_mean = df[['Rx', 'Ry', 'Rz']].mean(axis=1).as_matrix()
    axes[0].plot(np.arange(N), t_mean)
    axes[1].plot(np.arange(N), r_mean)
    axes[0].set_ylim(0, 0.08)
    axes[0].set_xlim(0, len(t_mean))
    axes[1].set_ylim(0, max_rotation)
    axes[1].set_xlim(0, len(t_mean))
    fig.savefig(os.path.join(figure_path, "Mean_Diff_{}.svg".format(name)), transparent=False)
    """

    # Draw poses
    column_tags = df_gt.columns.values
    numpy_gt = df_gt.as_matrix()
    numpy_prediction = df_pred.as_matrix()

    # save data
    figure_path = os.path.join(path, "figs")
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    plot_2d(numpy_gt, numpy_prediction, column_tags, "pose")
    fig.savefig(os.path.join(figure_path, "Sequence_Diff.png"))