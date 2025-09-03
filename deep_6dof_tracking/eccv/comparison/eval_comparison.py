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
    result_path = "/media/ssd/eccv/Results/result_dataframe"
    """
    root_path = "/media/ssd/eccv/Results/lowgeometry_results"
    models = ["single_low/clock", "single_low/dragon", "single_low/shoe", "single_low/skull",
              "single/clock", "single/shoe", "single/dragon", "single/skull",
              "general", "general_low"]
    output_name = "other_low"
    """

    # """
    root_path = "/media/ssd/eccv/Results/rebut_result"
    models = ["sqz/clock", "sqz/cookiejar", "sqz/dog", "sqz/dragon", "sqz/lego", "sqz/shoe",
              "sqz/skull", "sqz/walkman", "sqz/wateringcan", "sqz/turtle", "sqz/kinect",
              "res/clock", "res/cookiejar", "res/dog", "res/dragon", "res/lego", "res/shoe",
              "res/skull", "res/walkman", "res/wateringcan", "res/turtle", "res/kinect",
              "conv/lego", "conv/dog", "conv/clock", "conv/dragon", "conv/skull", "conv/shoe", "conv/walkman", "conv/wateringcan", "conv/cookiejar",
              #"conv/clock", "conv/cookiejar", "conv/dog", "conv/dragon", "conv/lego", "conv/shoe",
              #"conv/skull", "conv/walkman", "conv/wateringcan", "conv/turtle", "conv/kinect",
              #"conv/dog", "conv/cookiejar",
              "random_forest", "multi30_part_res", "generic"]
    output_name = "rebut"
    # """
    results = pd.DataFrame(columns=['object', 'sequence', 'speed_t', 'speed_r', 'speed_gt_t', 'speed_gt_r',
                                    'diff_t', 'diff_r', 'frame_id', 'lost_frame'])


    ############################################################################################################
    # Compute errors
    ############################################################################################################
    for model in models:
        sequences = [os.path.join(root_path, model, x) for x in os.listdir(os.path.join(root_path, model))]
        if sequences == []:
            print("[Warn] : Skip {}".format(model))
        for sequence in tqdm(sequences):
            # list of every word in sequence name
            # Warning, will bug if name has more than 1 _
            sequence_values = sequence.split("/")[-1].split("_")
            object = sequence_values[0]
            sequence_name = "_".join(sequence_values[1:])
            model_name = model
            if "/" in model:
                model_name = sequence.split("/")[-3]
                object = sequence.split("/")[-2]
                sequence_name = sequence.split("/")[-1]

            gt_path = os.path.join(sequence, "ground_truth_pose.csv")
            prediction_path = os.path.join(sequence, "prediction_pose.csv")
            if not os.path.exists(prediction_path):
                print("[Warn] Skip {}".format(prediction_path))
                continue
            magnitude_t, magnitude_r = eval_stability(prediction_path)
            speed_gt_t, speed_gt_r = eval_stability(gt_path)
            pose_diff_t, pose_diff_r = eval_pose_diff(gt_path, prediction_path)
            lost_frames = eval_tracking_loss(pose_diff_t, pose_diff_r)

            pose_diff_t *= 1000
            magnitude_t *= 1000
            speed_gt_t *= 1000

            sequence_results = pd.DataFrame(index=np.arange(len(magnitude_t)),
                                            columns=['model', 'object', 'sequence', 'speed_t', 'speed_r',
                                                     'diff_t', 'diff_r', 'frame_id', 'speed_gt_t', 'speed_gt_r',
                                                     'lost_frame'])
            for i in range(len(magnitude_t)):
                sequence_results.loc[i] = {"model": model_name, "object": object, "sequence": sequence_name,
                                           "speed_t": magnitude_t[i], "speed_r": magnitude_r[i],
                                           "diff_t": pose_diff_t[i], "diff_r": pose_diff_r[i],
                                           "frame_id": i, 'speed_gt_t': speed_gt_t[i], 'speed_gt_r': speed_gt_r[i],
                                           'lost_frame': lost_frames[i]}
            results = results.append(sequence_results)

    results = results.apply(pd.to_numeric, errors='ignore')
    results.to_csv(os.path.join(result_path, "{}.csv".format(output_name)))

