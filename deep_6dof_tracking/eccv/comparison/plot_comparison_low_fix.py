import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
font_size=22
sns.set_style("whitegrid")

ESTIMATOR = np.median


def fix_paper(df, legend_title, palette="Blues", filename=None, model_order=None):

    df = df[df['sequence'].str.contains("fix")]
    df.loc[df['sequence'].str.contains("near"), 'sequence'] = "near"
    df.loc[df['sequence'].str.contains("far"), 'sequence'] = "far"
    df.loc[df['sequence'].str.contains("occluded"), 'sequence'] = "occluded"
    order = ["near", "far", "occluded"]
    delta_t = 0.055
    df.speed_t /= delta_t
    df.speed_r /= delta_t

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    sns.boxplot(ax=ax, x="sequence", y="speed_t", data=df, palette=palette, hue="model", order=order,
                  showfliers=False, hue_order=model_order)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set(xlabel="", ylabel='Translation speed (mm/s)')
    plt.ylim([0, 4/delta_t])
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_T.pdf")

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    g = sns.boxplot(ax=ax, x="sequence", y="speed_r", data=df, palette=palette, hue="model", order=order,
                  showfliers=False, hue_order=model_order)
    ax.set(xlabel="", ylabel='Rotation speed (degree/s)')
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    plt.ylim([0, 4/delta_t])
    #plt.suptitle("Stability")
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")
    plt.show()

if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/University/Redaction/Publication/2018/ECCV/ECCV2018_DeepTrack/Images/Evaluation"

    dataframe = pd.read_csv(os.path.join(root_path, "other_low.csv"))
    dataframe = dataframe[dataframe['sequence'].str.contains("fix") == True]
    dataframe = dataframe[dataframe['frame_id'] > 15]
    dataframe = dataframe[dataframe['object'].str.contains("turtle") == False]

    blue_colors = sns.color_palette("Blues")
    red_colors = sns.color_palette("Reds")
    green_colors = sns.color_palette("Greens")
    yellow_colors = sns.color_palette("BrBG")
    colors = [blue_colors[-2], yellow_colors[0], blue_colors[-4], yellow_colors[1]]

    dataframe = dataframe.replace("single", "Ours object-specific")
    dataframe = dataframe.replace("general", "Ours generic")
    dataframe = dataframe.replace("single_low", "Low resolution object-specific")
    dataframe = dataframe.replace("general_low", "Low resolution generic")
    model_order = ["Ours object-specific", "Low resolution object-specific", "Ours generic", "Low resolution generic"]

    fix_paper(dataframe, "", filename=os.path.join(output_path, "comparison_stability_lowres"), palette=colors,
              model_order=model_order)


