import pandas as pd
import numpy as np
from deep_6dof_tracking.utils.angles import euler2mat
from deep_6dof_tracking.utils.transform import Transform
#from statsmodels import robust


def get_pose_difference(prediction, ground_truth):
    prediction_params = prediction.to_parameters(isDegree=True)
    ground_truth_params = ground_truth.to_parameters(isDegree=True)
    rotation = Transform()
    rotation[0:3, 0:3] = prediction[0:3, 0:3].dot(ground_truth[0:3, 0:3].transpose())
    difference = np.zeros(6)
    difference[3:] = np.abs(rotation.to_parameters(isDegree=True)[3:])
    difference[:3] = np.abs(prediction_params[:3] - ground_truth_params[:3])
    return difference


# def stability_evaluation(data):

#     std = data.std(axis=0)
#     mad = robust.mad(data, axis=0)
#     shifted_data = np.roll(data, 1, axis=0)
#     speed = np.mean(np.abs(shifted_data[1:-1] - data[1:-1]), axis=0)

#     return [std, mad, speed]


def get_pose_differences(prediction_df, gt_df):
    diffs = []
    for pred, gt in zip(prediction_df.as_matrix(), gt_df.as_matrix()):
        # handle the case where the parameters are eulers
        if len(pred) == 6:
            pred_transform = Transform.from_parameters(*pred)
            gt_transform = Transform.from_parameters(*gt)
        # handle the casee where the parameters are a matrix
        else:
            pred_transform = Transform.from_matrix(pred.reshape(4, 4))
            gt_transform = Transform.from_matrix(gt.reshape(4, 4))
        diff = get_pose_difference(pred_transform, gt_transform)
        diffs.append(diff)
    return np.array(diffs)


def eval_pose_diff(gt_path, prediction_path):
    df_gt = pd.read_csv(gt_path)
    df_pred= pd.read_csv(prediction_path)

    pose_diff = get_pose_differences(df_pred, df_gt)
    pose_diff_t, pose_diff_r = compute_pose_diff(pose_diff)
    return pose_diff_t, pose_diff_r


def compute_pose_diff(pose_diff):
    pose_diff_t = np.sqrt(np.square(pose_diff[:, 0]) + np.square(pose_diff[:, 1]) + np.square(pose_diff[:, 2]))
    #print(pose_diff_t)
    angle_diff = np.radians(pose_diff[:, 3:])
    pose_diff_r = np.zeros(len(pose_diff))
    # 0.002 sec
    for i in range(len(pose_diff)):
        mat = euler2mat(angle_diff[i, 0], angle_diff[i, 1], angle_diff[i, 2])
        pose_diff_r[i] = np.degrees(np.arccos((np.trace(mat) - 1) / 2))

        # pose_diff_t = np.mean(pose_diff[:, :3], axis=1)
        # pose_diff_r = np.mean(pose_diff[:, 3:], axis=1)
    return pose_diff_t, pose_diff_r


def eval_stability(prediction_path):
    df_pred = pd.read_csv(prediction_path).as_matrix()
    if len(df_pred[0]) != 6:
        converted_pred = []
        for prediction in df_pred:
            converted_pred.append(Transform.from_matrix(prediction.reshape(4, 4)).to_parameters())
        df_pred = np.array(converted_pred)

    shited = np.roll(df_pred, -1, axis=0)
    speed = (shited - df_pred)

    magnitude_t = np.sqrt(speed[:, 0] ** 2 + speed[:, 1] ** 2 + speed[:, 2] ** 2)
    magnitude_r = np.zeros(len(speed))
    # 0.002 sec
    for i in range(len(speed)):
        mat = euler2mat(speed[i, 3], speed[i, 4], speed[i, 5])
        magnitude_r[i] = np.degrees(np.arccos((np.trace(mat) - 1) / 2))
    return magnitude_t, magnitude_r


def eval_tracking_loss(diff_t, diff_r, max_rotation=20, max_translation=0.03, max_counter=7):
    counter = 0
    lost_frames = np.zeros(len(diff_t))
    for i, (t_err, r_err) in enumerate(zip(diff_t, diff_r)):

        error_detected = False
        if t_err > max_translation:
            error_detected = True
        if r_err > max_rotation:
            error_detected = True

        counter = counter + 1 if error_detected else 0

        if counter > max_counter:
            lost_frames[i] = 1
            counter = 0
    return lost_frames