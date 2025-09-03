import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from deep_6dof_tracking.eccv.eval_functions import eval_pose_diff, get_pose_differences
from deep_6dof_tracking.utils.transform import Transform


def matrix2euler(df):
    prediction_dict = {"Tx": [], "Ty": [], "Tz": [], "Rx": [], "Ry": [], "Rz": []}
    for prediction in df.as_matrix():
        trans = Transform.from_matrix(prediction.reshape(4, 4)).to_parameters(isDegree=True)
        prediction_dict["Tx"].append(trans[0])
        prediction_dict["Ty"].append(trans[1])
        prediction_dict["Tz"].append(trans[2])
        prediction_dict["Rx"].append(trans[3])
        prediction_dict["Ry"].append(trans[4])
        prediction_dict["Rz"].append(trans[5])
    prediction_dict["Tx"] = np.array(prediction_dict["Tx"])
    prediction_dict["Ty"] = np.array(prediction_dict["Ty"])
    prediction_dict["Tz"] = np.array(prediction_dict["Tz"])
    prediction_dict["Rx"] = np.array(prediction_dict["Rx"])
    prediction_dict["Ry"] = np.array(prediction_dict["Ry"])
    prediction_dict["Rz"] = np.array(prediction_dict["Rz"])
    return prediction_dict


if __name__ == '__main__':

    object = "kinect"
    sequence = "fix_near_1_cal"
    sequence2 = "fix_near_1"

    name = "res"
    name2 = "final"

    path = "/media/ssd/eccv/Results/test_results/{}/{}/{}".format(name, object, sequence)
    path2 = "/media/ssd/eccv/Results/test_results/{}/{}/{}".format(name, object, sequence2)

    gt_path = os.path.join(path, "ground_truth_pose.csv")
    prediction_path_ours = os.path.join(path, "prediction_pose.csv")
    prediction_path_them = os.path.join(path2, "prediction_pose.csv")

    predictions_ours = pd.read_csv(prediction_path_ours)
    predictions_them = pd.read_csv(prediction_path_them)
    gt = pd.read_csv(os.path.join(path, "ground_truth_pose.csv"))

    X = np.arange(0, len(gt))
    labels = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
    if len(predictions_ours.iloc[0]) == 16:
        predictions_ours = matrix2euler(predictions_ours)
        predictions_them = matrix2euler(predictions_them)
    if len(gt.iloc[0]) == 4*4:
        gt = matrix2euler(gt)
        X = np.arange(0, len(gt["Tx"]))

    for i, label in enumerate(labels):
        ax = plt.subplot(2, 3, i+1)
        plt.plot(X, predictions_ours[label], label=name)
        plt.plot(X, predictions_them[label], label=name2)
        plt.plot(X, gt[label], label="Gt")
        ax.legend(title=label)
    plt.show()

    labels = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
    for i, label in enumerate(labels):
        ax = plt.subplot(2, 3, i + 1)
        plt.plot(X, np.abs(predictions_ours[label] - gt[label]), label=name)
        plt.plot(X, np.abs(predictions_them[label] - gt[label]), label=name2)
        ax.legend(title=label)
    plt.show()

    ours_diff_t, ours_diff_r = eval_pose_diff(gt_path, prediction_path_ours)
    them_diff_t, them_diff_r = eval_pose_diff(gt_path, prediction_path_them)


    ax = plt.subplot("121")
    plt.plot(X, ours_diff_t, label="ours magnitude")
    plt.plot(X, them_diff_t, label="them magnitude")
    ax.legend(title="diff T")

    ax = plt.subplot("122")
    plt.plot(X, ours_diff_r, label="ours magnitude")
    plt.plot(X, them_diff_r, label="them magnitude")
    ax.legend(title="diff R")
    #plt.show()

    print("mean")
    print("{} T".format(name), np.mean(ours_diff_t, axis=0))
    print("{} T".format(name2), np.mean(them_diff_t, axis=0))

    print("{} R".format(name), np.mean(ours_diff_r, axis=0))
    print("{} R".format(name2), np.mean(them_diff_r, axis=0))

    print("median")
    print("{} T".format(name), np.median(ours_diff_t, axis=0))
    print("{} T".format(name2), np.median(them_diff_t, axis=0))

    print("{} R".format(name), np.median(ours_diff_r, axis=0))
    print("{} R".format(name2), np.median(them_diff_r, axis=0))

    print("RMSE")
    print("{} T".format(name), np.sqrt(np.mean(ours_diff_t**2, axis=0)))
    print("{} T".format(name2), np.sqrt(np.mean(them_diff_t**2, axis=0)))

    print("{} R".format(name), np.sqrt(np.mean(ours_diff_r**2, axis=0)))
    print("{} R".format(name2), np.sqrt(np.mean(them_diff_r**2, axis=0)))
    """
    diff_ours = get_pose_differences(predictions_ours, gt)
    diff_them = get_pose_differences(predictions_them, gt)
    print(diff_ours)
    ax = plt.subplot("121")
    plt.plot(X, np.mean(diff_ours[:, :3], axis=1), label="ours mean")
    plt.plot(X, np.mean(diff_them[:, :3], axis=1), label="them mean")
    ax.legend(title="diff T")

    ax = plt.subplot("122")
    plt.plot(X, np.mean(diff_ours[:, 3:], axis=1), label="ours mean")
    plt.plot(X, np.mean(diff_them[:, 3:], axis=1), label="them mean")
    ax.legend(title="diff R")

    plt.show()
    """