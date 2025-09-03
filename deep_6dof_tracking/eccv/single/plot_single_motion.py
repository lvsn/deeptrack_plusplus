import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
font_size = 22

sns.set_style("whitegrid")

ESTIMATOR = np.median

kind = "bin"


def clean_dataframe(df):
    df = df[df['sequence'].str.contains("motion")]
    df = df[df.diff_r != 0.0]
    df = df[df['object'].str.contains("turtle") == False]
    return df


def paper_plot(df, legend_title, filename=None, max_t=(40, 30), max_r=(16, 18), palette="Blues"):
    plt.figure(figsize=(12, 8))
    #delta_t = 0.055
    #df.speed_gt_t /= delta_t
    #df.speed_gt_r /= delta_t

    ax = plt.subplot("111")
    bins = np.linspace(0, int(max_t[0]) + 3, 5)
    df['binned'] = pd.cut(df['speed_gt_t'], bins)
    sns.boxplot(x="binned", y="diff_t", data=df, palette=palette, showfliers=False, hue="model")
    ax.set_ylim([0, max_t[1]])
    leg = ax.legend(loc="upper right", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)

    ax.set_xlabel("Translation speed (mm/frame)")
    ax.set_ylabel("Translation error (mm)")

    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_T.pdf")

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    bins = np.linspace(0, int(max_r[0]), 5)
    df['binned'] = pd.cut(df['speed_gt_r'], bins)
    sns.boxplot(x="binned", y="diff_r", data=df, palette=palette, showfliers=False, hue="model")
    ax.set_ylim([0, max_r[1]])
    leg = ax.legend(loc="upper right", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)

    ax.set_xlabel("Rotation speed (deg/frame)")
    ax.set_ylabel("Rotation error (deg)")

    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")
    plt.show()


def split_sequence_plot(df, save_path, legend_title, name, palette="Blues"):

    df_tmp = df[df['sequence'].str.contains("hard") == False]
    paper_plot(df_tmp, legend_title, os.path.join(save_path, "single_motion_" + name), palette=palette)

    df_tmp = df[df['sequence'].str.contains("hard")]
    model_list = set(df['model'].tolist())
    for model in model_list:
        lost_frames = df_tmp[df_tmp['model'].str.contains(model)].lost_frame.sum()
        print(model, lost_frames)
    #print(df_tmp[df['sequence'].str.contains("hard")])

if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/single_result_dataframe"
    object = "clock"

    df_translation = pd.read_csv(os.path.join(root_path, "single_translation.csv"))
    df_rotation = pd.read_csv(os.path.join(root_path, "single_rotation.csv"))
    df_bb = pd.read_csv(os.path.join(root_path, "single_bb.csv"))
    df_resolution = pd.read_csv(os.path.join(root_path, "single_resolution.csv"))

    df_translation = clean_dataframe(df_translation)
    df_rotation = clean_dataframe(df_rotation)
    df_bb = clean_dataframe(df_bb)
    df_resolution = clean_dataframe(df_resolution)

    df_translation = df_translation.replace("t1r20", "10")
    df_translation = df_translation.replace("t2r20", "20")
    df_translation = df_translation.replace("t3r20", "30")
    df_translation = df_translation.replace("t4r20", "40")
    df_translation = df_translation.replace("t5r20", "50")


    df_rotation = df_rotation.replace("t3r15", "15")
    df_rotation = df_rotation.replace("t3r20", "20")
    df_rotation = df_rotation.replace("t3r25", "25")
    df_rotation = df_rotation.replace("t3r30", "30")
    df_rotation = df_rotation.replace("t3r35", "35")

    names = ["15", "20", "25", "30", "35"]
    max_t = (40, 30)
    max_r = (16, 18)
    bins = np.linspace(0, int(max_t[0]) + 3, 5)
    df_rotation['binned_t'] = pd.cut(df_rotation['speed_gt_t'], bins)
    bins = np.linspace(0, int(max_r[0]), 5)
    df_rotation['binned_r'] = pd.cut(df_rotation['speed_gt_r'], bins)

    sequences = list(df_rotation.binned_t.unique())
    sequences.remove(np.nan)
    matrice_t = np.zeros((len(sequences), len(names)))
    medians = df_rotation.groupby(['binned_t', 'model'])['diff_t'].median().to_dict()
    print(medians)
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_t[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_t[i, j] = -1

    sequences = list(df_rotation.binned_r.unique())
    sequences.remove(np.nan)
    matrice_r = np.zeros((len(sequences), len(names)))
    medians = df_rotation.groupby(['binned_r', 'model'])['diff_r'].median().to_dict()
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
        total_string += (string_t + string_r) + "\\\\ \n"
        string_t = ""
        string_r = ""
    print(total_string)

    df_bb = df_bb.replace("bb0", "0%")
    df_bb = df_bb.replace("t3r20", "10%")
    df_bb = df_bb.replace("bb25", "25%")
    df_bb = df_bb.replace("bbm10", "-10%")
    df_bb = df_bb.replace("bbm25", "-25%")

    df_resolution = df_resolution.replace("r124", "124x124")
    df_resolution = df_resolution.replace("t3r20", "150x150")
    df_resolution = df_resolution.replace("r174", "174x174")
    df_resolution = df_resolution.replace("r200", "200x200")
    df_resolution = df_resolution.replace("r224", "224x224")
    df_resolution = df_resolution[df_resolution['model'].str.contains("224x224") == False]

    legend_title = "Translation Range (mm)"
    split_sequence_plot(df_translation, output_path, legend_title, "translation")
    legend_title = "Rotation Range (degree)"
    split_sequence_plot(df_rotation, output_path, legend_title, "rotation")
    legend_title = "Bounding box width"
    split_sequence_plot(df_bb, output_path, legend_title, "bb")
    legend_title = "Resolution"
    split_sequence_plot(df_resolution, output_path, legend_title, "res")








