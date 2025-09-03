import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

font_size = 22
sns.set_style("whitegrid")

ESTIMATOR = np.median


def fix_paper(df, legend_title, palette="Blues", filename=None, model_order=None):
    # delta_t = 0.055
    # df.speed_t /= delta_t
    # df.speed_r /= delta_t

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    sns.boxplot(ax=ax, x="sequence", y="speed_t", data=df, palette=palette, hue="model", order=order,
                showfliers=False, hue_order=model_order)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set(xlabel="", ylabel='Translation speed (mm/frame)')
    # plt.ylim([0, 4/delta_t])
    plt.ylim([0, 4])
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
    ax.set(xlabel="", ylabel='Rotation speed (degree/frame)')
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    # plt.ylim([0, 4/delta_t])
    plt.ylim([0, 4])
    # plt.suptitle("Stability")
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")

    g = sns.boxplot(ax=ax, x="sequence", y="relative_t", data=df, palette=palette, hue="model", order=order,
                    showfliers=False, hue_order=model_order)
    ax.set(xlabel="", ylabel='Relative Translation speed (%)')
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    # plt.ylim([0, 4/delta_t])
    plt.ylim([0, 0.02])
    # plt.suptitle("Stability")
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/University/Redaction/Publication/2018/ECCV/ECCV2018_DeepTrack/Images/Evaluation"
    file_name = "rebut.csv"

    object_sizes = {
        "clock": 221.97869420051575,
        "cookiejar": 186.63646280765533,
        "dog": 186.64510548114777,
        "dragon": 207.36300945281982,
        "lego": 118.56909096240997,
        "shoe": 313.98895382881165,
        "skull": 217.6094800233841,
        "walkman": 140.69706201553345,
        "wateringcan": 287.0884835720062,
        "turtle": 225.25633871555328,
        "kinect": 287.05358505249023
    }

    dataframe = pd.read_csv(os.path.join(root_path, file_name))
    dataframe = dataframe[dataframe['sequence'].str.contains("fix") == True]
    dataframe = dataframe[dataframe['frame_id'] > 20]

    blue_colors = sns.color_palette("Blues")
    red_colors = sns.color_palette("Reds")
    green_colors = sns.color_palette("Greens")
    colors = [blue_colors[-2], blue_colors[-3], blue_colors[-4], green_colors[-2], red_colors[-2]]

    df = dataframe[dataframe['sequence'].str.contains("fix")]
    df.loc[df['sequence'].str.contains("near"), 'sequence'] = "near"
    df.loc[df['sequence'].str.contains("far"), 'sequence'] = "far"
    df.loc[df['sequence'].str.contains("occluded"), 'sequence'] = "occluded"
    order = ["near", "far", "occluded"]

    sequences = ["near", "far", "occluded"]
    names = ["Ours specific", "Ours multi-object", "Ours generic", "Garon and Lalonde [1]", "Tan et al [5]"]
    models = ["res", "multi30_part_res", "generic", "conv", "random_forest"]

    for name, model in zip(names, models):
        df = df.replace(model, name)

    model_order = names

    # the speed is relative to the object size (%)
    for object_name, object_size in object_sizes.items():
        df["relative_t"] = df.loc[df['object'].str.contains(object_name), "speed_t"] / object_size

    matrice_r = np.zeros((len(sequences), len(names)))
    medians = df.groupby(['sequence', 'model'])['speed_r'].median().to_dict()

    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_r[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_r[i, j] = -1

    matrice_t = np.zeros((len(sequences), len(names)))
    medians = df.groupby(['sequence', 'model'])['speed_t'].median().to_dict()
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_t[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_t[i, j] = -1

    string_r = ""
    string_t = ""
    total_string = ""
    for j, model in enumerate(models):
        string_t += names[j]
        for i, sequence in enumerate(sequences):
            string_t += " & {:.2f}".format(matrice_t[i, j])
            string_r += " & {:.2f}".format(matrice_r[i, j])
        total_string += (string_t + string_r) + " \\\\ \n"
        string_t = ""
        string_r = ""
    print(total_string)
    fix_paper(df, "", filename=os.path.join(output_path, "comparison_stability"),
              palette=colors,
              model_order=model_order
              )
