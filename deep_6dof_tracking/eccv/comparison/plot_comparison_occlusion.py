import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
font_size=22
sns.set_style("whitegrid")

ESTIMATOR = np.median


def occlusion_model_plots(dataframe, palette="Blues", legend_title=None, filename=None, model_order=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    sns.boxplot(x="sequence", y="diff_t", hue="model", data=dataframe, palette=palette,
                showfliers=False, hue_order=model_order)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set_ylabel("Translation error (mm)")
    ax.set_xlabel("Occlusion %")
    ax.set_ylim([0, 85])
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
    sns.boxplot(x="sequence", y="diff_r", hue="model", data=dataframe, palette=palette, showfliers=False,
                hue_order=model_order)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set_ylabel("Rotation error (deg)")
    ax.set_xlabel("Occlusion %")
    ax.set_ylim([0, 60])
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize='17')  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")
    plt.show()


def replace_occlusion_sequence_names(df, post=""):
    df = df.replace("occlusion_0{}".format(post), "0")
    df = df.replace("occlusion_v_15{}".format(post), "15")
    df = df.replace("occlusion_v_30{}".format(post), "30")
    df = df.replace("occlusion_v_45{}".format(post), "45")
    df = df.replace("occlusion_v_60{}".format(post), "60")
    df = df.replace("occlusion_v_75{}".format(post), "75")

    df = df.replace("occlusion_h_15{}".format(post), "15")
    df = df.replace("occlusion_h_30{}".format(post), "30")
    df = df.replace("occlusion_h_45{}".format(post), "45")
    df = df.replace("occlusion_h_60{}".format(post), "60")
    df = df.replace("occlusion_h_75{}".format(post), "75")
    return df


def clean_dataframe(df):
    df = df[df['sequence'].str.contains("occlusion")]
    df = df[df.diff_r != 0]
    df = replace_occlusion_sequence_names(df)
    #df = df[df['object'].str.contains("turtle") == False]
    #df = df[df['object'].str.contains("skull") == False]
    #df = df[df['object'].str.contains("dragon") == False]

    return df

if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/University/Redaction/Publication/2018/ECCV/ECCV2018_DeepTrack/Images/Evaluation"

    dataframe = pd.read_csv(os.path.join(root_path, "rebut.csv"))

    dataframe = clean_dataframe(dataframe)

    blue_colors = sns.color_palette("Blues")
    red_colors = sns.color_palette("Reds")
    green_colors = sns.color_palette("Greens")
    colors = [blue_colors[-2], blue_colors[-3], blue_colors[-4], green_colors[-2], red_colors[-2]]

    #dataframe = dataframe.replace("random_forest", "Tan et al. 2015")
    #dataframe = dataframe.replace("multi29_notpart_res", "Ours generic")
    #dataframe = dataframe.replace("multi30_part", "Ours multi-object")
    #dataframe = dataframe.replace("conv", "Garon and Lalonde 2017")
    #dataframe = dataframe.replace("single", "Ours object-specific")
    #model_order = ["Ours object-specific", "Ours multi-object", "Ours generic", "Garon and Lalonde 2017", "Tan et al. 2015"]


    sequences = dataframe.sequence.unique()
    sequences.sort()
    names = ["Ours specific", "Ours multi-object", "Ours generic", "Garon and Lalonde [1]", "Tan et al [5]"]
    models = ["res", "multi30_part_res", "generic", "conv", "random_forest"]

    for name, model in zip(names, models):
        dataframe = dataframe.replace(model, name)

    model_order = names

    occlusion_model_plots(dataframe,
                          palette=colors,
                          model_order=model_order,
                          filename=os.path.join(output_path, "comparison_occlusion")
                          )

    matrice_t = np.zeros((len(sequences), len(names)))
    medians = dataframe.groupby(['sequence', 'model'])['diff_t'].median().to_dict()
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_t[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_t[i, j] = -1

    matrice_r = np.zeros((len(sequences), len(names)))
    medians = dataframe.groupby(['sequence', 'model'])['diff_r'].median().to_dict()
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_r[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_r[i, j] = -1


    string_r = ""
    string_t = ""
    total_string = ""
    for j, model in enumerate(names):
        string_t += names[j]
        for i, sequence in enumerate(sequences):
            string_t += " & {:.1f}".format(matrice_t[i, j])
            string_r += " & {:.1f}".format(matrice_r[i, j])
        total_string += (string_t + string_r) + " \\\\ \n"
        string_t = ""
        string_r = ""
    print(total_string)

