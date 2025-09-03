import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

ESTIMATOR = np.median


def compare_part_vs_notpart(df_not_occluded, df_occluded):
    df_not_occluded = df_not_occluded[df_not_occluded['training'].str.contains("mse") == True]
    df_occluded = df_occluded[df_occluded['training'].str.contains("mse") == True]

    df_total = df_not_occluded.append(df_occluded)
    # Compare x object and geo vs squeeze
    order = [1, 5, 10, 20, 26]
    hue = "is_part"

    ax = plt.subplot("321")
    sns.violinplot(x="model", y="speed_t", data=df_not_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])
    ax = plt.subplot("322")
    sns.violinplot(x="model", y="speed_r", data=df_not_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])

    ax = plt.subplot("323")
    sns.violinplot(x="model", y="speed_t", data=df_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])
    ax = plt.subplot("324")
    sns.violinplot(x="model", y="speed_r", data=df_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])

    ax = plt.subplot("325")
    sns.violinplot(x="model", y="speed_t", data=df_total, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])
    ax = plt.subplot("326")
    sns.violinplot(x="model", y="speed_r", data=df_total, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])

    plt.show()


def compare_projection_vs_mse(df_not_occluded, df_occluded, show_ispart=True):
    df_not_occluded = df_not_occluded[df_not_occluded['is_part'] == show_ispart]
    df_occluded = df_occluded[df_occluded['is_part'] == show_ispart]
    df_total = df_not_occluded.append(df_occluded)
    # Compare x object and geo vs squeeze
    order = [1, 5, 10, 20, 26]
    hue = "training"

    ax = plt.subplot("321")
    sns.violinplot(x="model", y="speed_t", data=df_not_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])
    ax = plt.subplot("322")
    sns.violinplot(x="model", y="speed_r", data=df_not_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])

    ax = plt.subplot("323")
    sns.violinplot(x="model", y="speed_t", data=df_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])
    ax = plt.subplot("324")
    sns.violinplot(x="model", y="speed_r", data=df_occluded, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])

    ax = plt.subplot("325")
    sns.violinplot(x="model", y="speed_t", data=df_total, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])
    ax = plt.subplot("326")
    sns.violinplot(x="model", y="speed_r", data=df_total, order=order,
                   hue=hue, split=True, inner="quart",
                   estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 5])

    plt.show()


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/result_dataframe"

    single_fix = pd.read_csv(os.path.join(root_path, "single_translation.csv"))
    dataframe_multi = pd.read_csv(os.path.join(root_path, "multi.csv"))

    dataframe_multi = dataframe_multi[dataframe_multi['sequence'].str.contains("fix") == True]

    single_fix = single_fix[single_fix['frame_id'] > 15]
    dataframe_multi = dataframe_multi[dataframe_multi['frame_id'] > 15]

    # keep only t3r20 fix datasets
    dataframe_single = single_fix[single_fix['model'].str.contains("t3r20") == True]
    dataframe_single = dataframe_single.replace("t3r20", 1)
    dataframe_single.loc[:, 'is_part'] = pd.Series(True, index=dataframe_single.index)
    dataframe_single = dataframe_single[dataframe_single['sequence'].str.contains("fix") == True]

    # Remove turtle
    dataframe_single = dataframe_single[dataframe_single['object'].str.contains("turtle") == False]
    dataframe_multi = dataframe_multi[dataframe_multi['object'].str.contains("turtle") == False]

    df_single_not_occ = dataframe_single[dataframe_single['sequence'].str.contains("occluded") == False]
    df_single_occ = dataframe_single[dataframe_single['sequence'].str.contains("occluded") == True]
    df_multi_not_occ = dataframe_multi[dataframe_multi['sequence'].str.contains("occluded") == False]
    df_multi_occ = dataframe_multi[dataframe_multi['sequence'].str.contains("occluded") == True]

    df_not_occluded = df_multi_not_occ.append(df_single_not_occ)
    df_occluded = df_multi_occ.append(df_single_occ)

    compare_part_vs_notpart(df_not_occluded, df_occluded)
    compare_projection_vs_mse(df_not_occluded, df_occluded)



