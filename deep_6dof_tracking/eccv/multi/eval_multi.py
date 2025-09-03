import os

import math

from deep_6dof_tracking.eccv.eval_functions import eval_stability, eval_pose_diff
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
    root_path = "/media/ssd/eccv/final_results"
    result_path = "/media/ssd/eccv/result_dataframe"

    models = ["multi5_notpart", "multi5_notpart_geo", "multi5_part", "multi5_part_geo",
              "multi10_notpart_mse", "multi10_part", "multi10_part_geo",
              "multi20_notpart_mse", "multi20_part", "multi20_part_geo",
              "multi26_notpart_mse", "multi26_part", "multi26_part_geo"]

    #models = ["multi5_part", "multi10_part", "multi20_part", "multi26_part"]
    results = pd.DataFrame(columns=['object', 'sequence', 'speed_t', 'speed_r', 'speed_gt_t', 'speed_gt_r',
                                    'diff_t', 'diff_r',
                                    'training', 'is_part',
                                    'frame_id'])


    ############################################################################################################
    # Compute errors
    ############################################################################################################
    for model in models:
        sequences = [os.path.join(root_path, model, x) for x in os.listdir(os.path.join(root_path, model))]
        print("Eval model : {}".format(model))
        for sequence in tqdm(sequences):
            # list of every word in sequence name
            # Warning, will bug if name has more than 1 _
            sequence_values = sequence.split("/")[-1].split("_")
            object = sequence_values[0]
            sequence_name = "_".join(sequence_values[1:])

            gt_path = os.path.join(sequence, "ground_truth_pose.csv")
            prediction_path = os.path.join(sequence, "prediction_pose.csv")
            if not os.path.exists(prediction_path):
                print("[Warn] Skip {}".format(prediction_path))
                continue
            magnitude_t, magnitude_r = eval_stability(prediction_path)
            speed_gt_t, speed_gt_r = eval_stability(gt_path)
            pose_diff_t, pose_diff_r = eval_pose_diff(gt_path, prediction_path)

            pose_diff_t *= 1000
            magnitude_t *= 1000
            speed_gt_t *= 1000

            sequence_results = pd.DataFrame(index=np.arange(len(magnitude_t)),
                                            columns=['model', 'object', 'sequence', 'speed_t', 'speed_r',
                                                     'diff_t', 'diff_r', 'speed_gt_t', 'speed_gt_r',
                                                     'training', 'is_part', 'frame_id'])
            training = "mse"
            model_name = model
            if "geo" in model:
                training = "projection"
                model_name = model[:-4]
            if "mse" in model:
                training = "mse"
                model_name = model[:-4]
            model_split = model_name.split("_")
            model_name = model_split[0][5:]
            is_part = model_split[1] == "part"
            for i in range(len(magnitude_t)):
                sequence_results.loc[i] = {"model": model_name, "object": object, "sequence": sequence_name,
                                           "speed_t": magnitude_t[i], "speed_r": magnitude_r[i],
                                           "diff_t": pose_diff_t[i], "diff_r": pose_diff_r[i],
                                           "training": training, "is_part": is_part, 'frame_id': i,
                                           'speed_gt_t': speed_gt_t[i], 'speed_gt_r': speed_gt_r[i]}
            results = results.append(sequence_results)

    print("Saving file...")
    results = results.apply(pd.to_numeric, errors='ignore')
    results.to_csv(os.path.join(result_path, "multi.csv"))

