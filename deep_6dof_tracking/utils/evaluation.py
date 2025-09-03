import numpy as np
import pandas as pd
import os

from statsmodels import robust

from deep_6dof_tracking.utils.transform import Transform


def get_pose_difference(prediction, ground_truth):
    prediction_params = prediction.to_parameters(isDegree=True)
    ground_truth_params = ground_truth.to_parameters(isDegree=True)
    rotation = Transform()
    rotation[0:3, 0:3] = prediction[0:3, 0:3].dot(ground_truth[0:3, 0:3].transpose())
    difference = np.zeros(6)
    difference[3:] = np.abs(rotation.to_parameters(isDegree=True)[3:])
    difference[:3] = abs(prediction_params[:3] - ground_truth_params[:3])
    return difference


def stability_evaluation(data):

    std = data.std(axis=0)
    mad = robust.mad(data, axis=0)
    shifted_data = np.roll(data, 1, axis=0)
    speed = np.mean(np.abs(shifted_data[1:-1] - data[1:-1]), axis=0)

    return [std, mad, speed]