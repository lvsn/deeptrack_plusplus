import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

ESTIMATOR = np.median


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/result_dataframe"

    dataframe_multi = pd.read_csv(os.path.join(root_path, "multi.csv"))

    dataframe_multi = dataframe_multi[dataframe_multi['sequence'].str.contains("motion")]
    dataframe_multi = dataframe_multi[dataframe_multi['object'].str.contains("dragon")]
    dataframe_multi = dataframe_multi[dataframe_multi['training'].str.contains("mse")]
    dataframe_multi = dataframe_multi[dataframe_multi['sequence'].str.contains("motion_rotation")]
    dataframe_multi = dataframe_multi[dataframe_multi['is_part']]
    dataframe_multi = dataframe_multi[:-1]

    dataframe_multi_26 = dataframe_multi[dataframe_multi['model'] == 26]
    dataframe_multi_10 = dataframe_multi[dataframe_multi['model'] == 10]

    sns.jointplot(x="speed_gt_t", y="diff_t", data=dataframe_multi_26, kind="kde", color="k", xlim=(0, 0.025))
    sns.jointplot(x="speed_gt_t", y="diff_t", data=dataframe_multi_10, kind="kde", color="k", xlim=(0, 0.025))
    plt.show()

    sns.jointplot(x="speed_gt_r", y="diff_r", data=dataframe_multi_26, kind="kde", color="k", xlim=(0, 20))
    sns.jointplot(x="speed_gt_r", y="diff_r", data=dataframe_multi_10, kind="kde", color="k", xlim=(0, 20))
    plt.show()


