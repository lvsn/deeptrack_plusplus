import os

import math

from deep_6dof_tracking.eccv.eval_functions import eval_stability, eval_pose_diff, eval_tracking_loss
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from deep_6dof_tracking.utils.angles import euler2mat
from deep_6dof_tracking.utils.evaluation import stability_evaluation, get_pose_difference
from deep_6dof_tracking.utils.transform import Transform


def get_pose_differences(prediction_df, gt_df):
    diffs = []
    for pred, gt in zip(prediction_df.as_matrix(), gt_df.as_matrix()):
        pred_transform = Transform.from_parameters(*pred)
        gt_transform = Transform.from_parameters(*gt)
        diff = get_pose_difference(pred_transform, gt_transform)
        diffs.append(diff)
    return np.array(diffs)

if __name__ == '__main__':
    sequence_path = "/media/ssd/eccv/Results/rebut_result_choi/multi30_part_res"
    ############################################################################################################
    # Compute errors
    ############################################################################################################
    sequences = [os.path.join(sequence_path, x) for x in os.listdir(os.path.join(sequence_path))]
    total=np.zeros(6)
    for sequence in tqdm(sequences):
        gt_path = os.path.join(sequence, "ground_truth_pose.csv")
        prediction_path = os.path.join(sequence, "prediction_pose.csv")

        df_gt = pd.read_csv(gt_path)
        df_pred = pd.read_csv(prediction_path)
        diffs = np.zeros((len(df_pred), 6))
        count = 0
        for pred, gt in zip(df_pred.as_matrix(), df_gt.as_matrix()):
            pred_transform = Transform.from_matrix(pred.reshape(4, 4))
            gt_transform = Transform.from_matrix(gt.reshape(4, 4))
            diffs[count, :] = get_pose_difference(pred_transform, gt_transform)
            #print(list(diffs[count, :]))
            count += 1
        #diffs[:, 2] -= 0.01
        print(sequence.split("/")[-1])
        print(list(np.mean(diffs, axis=0)))
        #if sequence.split("/")[-1] != "milk":
        total += np.mean(diffs, axis=0)
        #print("Mean T {}".format(np.mean(diff[:3])))
        # print("Mean R {}".format(np.mean(diff[3:])))
    mean = total/4
    print(list(mean))
    print(np.mean(mean[3:]))
    print(np.mean(mean[:2]))

