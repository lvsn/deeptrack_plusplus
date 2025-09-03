import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
font_size=22

ESTIMATOR = np.median


def paper_plot(df, legend_title, order=None, filename=None, max_t=(50, 30), max_r=(75, 18), palette="Blues"):
    plt.figure(figsize=(12, 8))

    ax = plt.subplot("111")
    sns.boxplot(x="binned_t", y="diff_t", data=df, palette=palette, showfliers=False, hue="model", hue_order=order)
    ax.set_ylim([0, max_t[1]])
    leg = ax.legend(loc="upper right", title="", frameon=True, prop={'size': 14})
    leg.get_frame().set_alpha(1)

    ax.set_xlabel("Translation speed (mm/frame)")
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
    sns.boxplot(x="binned_relative_t", y="diff_t", data=df, palette=palette, showfliers=False, hue="model", hue_order=order)
    ax.set_ylim([0, max_t[1]])
    leg = ax.legend(loc="upper right", title="", frameon=True, prop={'size': 14})
    leg.get_frame().set_alpha(1)

    ax.set_xlabel("Relative translation (%)")
    ax.set_ylabel("Translation error (mm)")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    sns.boxplot(x="binned_r", y="diff_r", data=df, palette=palette, showfliers=False, hue="model", hue_order=order)
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

    dataframe = pd.read_csv(os.path.join(root_path, "rebut.csv"))

    dataframe2 = dataframe[dataframe['sequence'].str.contains("motion") == True]

    for object_name, object_size in object_sizes.items():
        dataframe2["relative_t"] = dataframe2.loc[dataframe2['object'].str.contains(object_name), "speed_gt_t"] / object_size

    #for object_name, object_size in object_sizes.items():
    #    dataframe2.loc[dataframe2['object'].str.contains(object_name), "relative_t"] /= object_size

    blue_colors = sns.color_palette("Blues")
    red_colors = sns.color_palette("Reds")
    green_colors = sns.color_palette("Greens")
    colors = [blue_colors[-2], blue_colors[-3], blue_colors[-4], green_colors[-2], red_colors[-2]]

    #dataframe2 = dataframe2.replace("random_forest", "Tan et al. 2015")
    #dataframe2 = dataframe2.replace("multi29_notpart_res", "Ours generic")
    #dataframe2 = dataframe2.replace("multi30_part", "Ours multi-object")
    #dataframe2 = dataframe2.replace("conv", "Garon and Lalonde 2017")
    #dataframe2 = dataframe2.replace("single", "Ours object-specific")
    #model_order = ["Ours object-specific", "Ours multi-object", "Ours generic", "Garon and Lalonde 2017",
    #               "Tan et al. 2015"]

    #dataframe_tmp = dataframe[dataframe['sequence'].str.contains("hard") == False]
    #dataframe_tmp = dataframe_tmp[dataframe_tmp.diff_r != 0]

    dataframe_tmp = dataframe2[dataframe2['sequence'].str.contains("hard") == False]
    dataframe_tmp = dataframe_tmp[dataframe_tmp.diff_r != 0]

    max_t = (50, 30)
    max_r = (75, 18)
    bins = np.linspace(0, max_t[0], 5)
    dataframe_tmp['binned_t'] = pd.cut(dataframe_tmp['speed_gt_t'], bins)
    bins = np.linspace(0, max_r[0], 5)
    dataframe_tmp['binned_r'] = pd.cut(dataframe_tmp['speed_gt_r'], bins)
    bins = np.linspace(0, 0.1, 5)
    dataframe_tmp['binned_relative_t'] = pd.cut(dataframe_tmp['relative_t'], bins)

    names = ["Ours specific", "Ours multi-object", "Ours generic", "Garon and Lalonde [1]", "Tan et al [5]"]
    models = ["res", "multi30_part_res", "generic", "conv", "random_forest"]

    for name, model in zip(names, models):
        dataframe_tmp = dataframe_tmp.replace(model, name)

    model_order = names

    paper_plot(dataframe_tmp, "Full", filename=os.path.join(output_path, "comparison_motion"),
               order=model_order,
               palette=colors
               )


    sequences = list(dataframe_tmp.binned_t.unique())
    sequences.remove(np.nan)
    matrice_t = np.zeros((len(sequences), len(names)))
    medians = dataframe_tmp.groupby(['binned_t', 'model'])['diff_t'].median().to_dict()
    print(medians)
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_t[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_t[i, j] = -1

    sequences = list(dataframe_tmp.binned_r.unique())
    sequences.remove(np.nan)
    matrice_r = np.zeros((len(sequences), len(names)))
    medians = dataframe_tmp.groupby(['binned_r', 'model'])['diff_r'].median().to_dict()
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_r[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_r[i, j] = -1

    fails = []
    dataframe_tmp = dataframe2[dataframe2['sequence'].str.contains("hard")]
    print(dataframe_tmp)
    for model in names:
        model_dataframe = dataframe_tmp[dataframe_tmp['model'] == model]
        lost_frames = model_dataframe.diff_r == 0.0
        print(len(lost_frames))
        print(model, lost_frames.sum())
        fails.append(lost_frames.sum())

    string_r = ""
    string_t = ""
    total_string = ""
    for j, model in enumerate(names):
        string_t += names[j]
        for i, sequence in enumerate(sequences):
            string_t += " & {:.1f}".format(matrice_t[i, j])
            string_r += " & {:.1f}".format(matrice_r[i, j])
        total_string += (string_t + string_r) + " & {} \\\\ \n".format(fails[j])
        string_t = ""
        string_r = ""
    print(total_string)





