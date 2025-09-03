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


if __name__ == '__main__':
    root_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/final_results"
    result_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/single_result_dataframe"

    ############################################################################################################
    # Experiments
    ############################################################################################################
    experiment_names = ["translation", "rotation", "bb", "resolution"]
    experiment1 = ["t1r20", "t2r20", "t3r20", "t4r20", "t5r20"]
    experiment2 = ["t3r15", "t3r20", "t3r25", "t3r30", "t3r35"]
    experiment3 = ["bbm25", "bbm10", "bb0", "t3r20", "bb25"]
    experiment4 = ["r124", "t3r20", "r174", "r200", "r224"]
    experiment_temp = ["t2r20-channel", "t2r20-old"]

    experiments = [experiment1, experiment2, experiment3, experiment4]
    #experiment_names = ["turtle"]
    #experiments = [experiment_temp]
    objects = ["dragon", "clock", "shoe", "skull"]

    ############################################################################################################
    # Compute errors
    ############################################################################################################

    for experiment_name, experiment in zip(experiment_names, experiments):
        #sequence_filters = ["occlusion"]
        #sequence_filters = []
        results = pd.DataFrame(columns=['model', 'object', 'sequence', 'speed_t', 'speed_r', 'training', 'speed_gt_t', 'speed_gt_r',
                                        'diff_t', 'diff_r', "frame_id", "lost_frame"])
        print("Evaluate {}".format(experiment))
        for i, object in tqdm(enumerate(objects), total=len(objects)):
            for j, model in enumerate(experiment):
                # The model has to exist for the object!
                object_model_path = os.path.join(root_path, object, model)
                if not os.path.exists(object_model_path):
                    continue

                # Compute for all sequences
                sequences = os.listdir(object_model_path)
                sequence_stability = []
                for sequence in sequences:
                    gt_path = os.path.join(object_model_path, sequence, "ground_truth_pose.csv")
                    prediction_path = os.path.join(object_model_path, sequence, "prediction_pose.csv")
                    if not os.path.exists(prediction_path):
                        print("[Warn] skip : {}".format(prediction_path))
                        continue
                    pose_diff_t, pose_diff_r = eval_pose_diff(gt_path, prediction_path)
                    magnitude_t, magnitude_r = eval_stability(prediction_path)
                    speed_gt_t, speed_gt_r = eval_stability(gt_path)
                    lost_frames = eval_tracking_loss(pose_diff_t, pose_diff_r)

                    # convert to mm
                    pose_diff_t *= 1000
                    magnitude_t *= 1000
                    speed_gt_t *= 1000

                    sequence_results = pd.DataFrame(index=np.arange(len(magnitude_t)), columns=['model', 'object',
                                                                                                'sequence', 'speed_t',
                                                                                                'speed_r', 'training',
                                                                                                'diff_t', 'diff_r',
                                                                                                'speed_gt_t', 'speed_gt_r',
                                                                                                "frame_id", "lost_frame"])

                    training = "mse"
                    model_name = model
                    if "geo" in model:
                        training = "projection"
                        model_name = model[:-4]

                    for i in range(len(magnitude_t)):
                        sequence_results.loc[i] = {"model": model_name, "object": object, "sequence": sequence,
                                                   "speed_t": magnitude_t[i], "speed_r": magnitude_r[i],
                                                   "diff_t": pose_diff_t[i], "diff_r": pose_diff_r[i],
                                                   "training": training, "lost_frame":lost_frames[i],
                                                   "frame_id": i, 'speed_gt_t': speed_gt_t[i],
                                                   'speed_gt_r': speed_gt_r[i]}
                    results = results.append(sequence_results)

        results = results.apply(pd.to_numeric, errors='ignore')
        results.to_csv(os.path.join(result_path, "single_{}.csv".format(experiment_name)))
