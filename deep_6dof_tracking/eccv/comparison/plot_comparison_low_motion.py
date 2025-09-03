import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
font_size=22

ESTIMATOR = np.median


def paper_plot(df, legend_title, order, filename=None, max_t=(40, 30), max_r=(16, 18), palette="Blues"):
    plt.figure(figsize=(12, 8))

    ax = plt.subplot("111")
    bins = np.linspace(0, max_t[0], 5)
    df['binned'] = pd.cut(df['speed_gt_t'], bins)
    sns.boxplot(x="binned", y="diff_t", data=df, palette=palette, showfliers=False, hue="model", hue_order=order)
    ax.set_ylim([0, max_t[1]])
    leg = ax.legend(loc="upper right", title="", frameon=True, prop={'size': 14})
    leg.get_frame().set_alpha(1)

    ax.set_xlabel("Translation speed (mm/s)")
    ax.set_ylabel("Translation error (mm)")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_T.pdf")

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    bins = np.linspace(0, max_r[0], 5)
    df['binned'] = pd.cut(df['speed_gt_r'], bins)
    sns.boxplot(x="binned", y="diff_r", data=df, palette=palette, showfliers=False, hue="model", hue_order=order)
    ax.set_ylim([0, max_r[1]])
    leg = ax.legend(loc="upper right", title="", frameon=True, prop={'size': 14})
    leg.get_frame().set_alpha(1)

    ax.set_xlabel("Rotation speed (deg/s)")
    ax.set_ylabel("Rotation error (deg)")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")
    plt.show()

if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/University/Redaction/Publication/2018/ECCV/ECCV2018_DeepTrack/Images/Evaluation"

    dataframe = pd.read_csv(os.path.join(root_path, "other_low.csv"))

    dataframe2 = dataframe[dataframe['sequence'].str.contains("motion") == True]
    #dataframe2 = dataframe2.append(dataframe[dataframe['sequence'].str.contains("motion") == True])
    dataframe2 = dataframe2[dataframe2['object'].str.contains("turtle") == False]
    #dataframe = dataframe[dataframe['object'].str.contains("skull") == False]

    blue_colors = sns.color_palette("Blues")
    red_colors = sns.color_palette("Reds")
    green_colors = sns.color_palette("Greens")
    yellow_colors = sns.color_palette("BrBG")
    colors = [blue_colors[-2], yellow_colors[0], blue_colors[-4], yellow_colors[1]]

    dataframe2 = dataframe2.replace("single", "Ours object-specific")
    dataframe2 = dataframe2.replace("general", "Ours generic")
    dataframe2 = dataframe2.replace("single_low", "Low resolution object-specific")
    dataframe2 = dataframe2.replace("general_low", "Low resolution generic")
    model_order = ["Ours object-specific", "Low resolution object-specific", "Ours generic", "Low resolution generic"]

    #dataframe_tmp = dataframe[dataframe['sequence'].str.contains("hard") == False]
    #dataframe_tmp = dataframe_tmp[dataframe_tmp.diff_r != 0]
    #paper_plot(dataframe2, "Full", order=model_order, palette=colors)

    dataframe_tmp = dataframe2[dataframe2['sequence'].str.contains("hard") == False]
    dataframe_tmp = dataframe_tmp[dataframe_tmp.diff_r != 0]
    paper_plot(dataframe_tmp, "Full", filename=os.path.join(output_path, "comparison_motion_lowres"), order=model_order, palette=colors)

    #dataframe_tmp = dataframe[dataframe['sequence'].str.contains("rotation") == False]
    #dataframe_tmp = dataframe_tmp[dataframe_tmp['sequence'].str.contains("translation") == False]
    dataframe_tmp = dataframe_tmp[dataframe_tmp['sequence'].str.contains("hard")]
    paper_plot(dataframe2, "Hard", order=model_order, palette=colors)

    dataframe_tmp = dataframe2[dataframe2['sequence'].str.contains("hard")]
    for model in model_order:
        model_dataframe = dataframe_tmp[dataframe_tmp['model'].str.contains(model)]
        lost_frames = model_dataframe.diff_r == 0.0
        print(len(lost_frames))
        print(model, lost_frames.sum())





