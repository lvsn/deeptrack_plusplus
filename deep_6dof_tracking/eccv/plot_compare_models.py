import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

ESTIMATOR = np.median

def fix_object_plots(df, total=1, n=1):
    ax = plt.subplot(total, 2, n)
    sns.pointplot(ax=ax, x="model", y="speed_t", data=df, hue="object", linestyles="--", scale=0.5, estimator=ESTIMATOR)
    sns.pointplot(ax=ax, x="model", y="speed_t", data=df, color="black", label="total")
    plt.ylim([0, 0.002])
    ax = plt.subplot(total, 2, n + 1)
    sns.pointplot(ax=ax, x="model", y="speed_r", data=df, hue="object", linestyles="--", scale=0.5, estimator=ESTIMATOR)
    sns.pointplot(ax=ax, x="model", y="speed_r", data=df, color="black", label="total")
    plt.ylim([0, 2])


def occlusion_model_plots(dataframe):
    dataframe = replace_occlusion_sequence_names(dataframe, "_reset15")

    ax = plt.subplot("121")
    sns.pointplot(x="sequence", y="diff_t", hue="model", data=dataframe, scale=0.75, palette="Blues", estimator=ESTIMATOR)
    ax.set_ylim([0, 0.06])

    ax = plt.subplot("122")
    sns.pointplot(x="sequence", y="diff_r", hue="model", data=dataframe, scale=0.75, palette="Blues", estimator=ESTIMATOR)
    ax.set_ylim([0, 35])
    plt.show()


def occlusion_object_plots(dataframe, model):
    dataframe = replace_occlusion_sequence_names(dataframe, "_reset15")
    dataframe = dataframe[dataframe['model'].str.contains(model) == True]

    ax = plt.subplot("121")
    sns.pointplot(x="sequence", y="diff_t", hue="object", data=dataframe, linestyles="--", scale=0.5, estimator=ESTIMATOR)
    sns.pointplot(x="sequence", y="diff_t", data=dataframe, color="black", label="total", estimator=ESTIMATOR)
    ax.set_ylim([0, 0.06])

    ax = plt.subplot("122")
    sns.pointplot(x="sequence", y="diff_r", hue="object", data=dataframe, linestyles="--", scale=0.5, estimator=ESTIMATOR)
    sns.pointplot(x="sequence", y="diff_r", data=dataframe, color="black", label="total", estimator=ESTIMATOR)
    ax.set_ylim([0, 35])
    plt.suptitle(model)
    plt.show()


def test_stability(dataframe):
    # fix without occluded
    fix_clear = dataframe[dataframe['sequence'].str.contains("occluded") == False]
    fix_object_plots(fix_clear, 3, 1)

    # fix occluded only
    fix_occluded = dataframe[dataframe['sequence'].str.contains("occluded") == True]
    fix_object_plots(fix_occluded, 3, 3)

    # all
    fix_object_plots(dataframe, 3, 5)

    plt.show()


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/result_dataframe"

    dataframe = pd.read_csv(os.path.join(root_path, "single_turtle.csv"))
    dataframe = dataframe[dataframe['sequence'].str.contains("fix") == True]

    ax = plt.subplot("121")
    sns.violinplot(ax=ax, x="model", y="speed_t", data=dataframe, estimator=ESTIMATOR)
    ax = plt.subplot("122")
    sns.violinplot(ax=ax, x="model", y="speed_r", data=dataframe, estimator=ESTIMATOR)
    plt.show()

