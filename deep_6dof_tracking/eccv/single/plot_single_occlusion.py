import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deep_6dof_tracking.eccv.comparison.plot_comparison_occlusion import replace_occlusion_sequence_names
font_size=22
sns.set_style("whitegrid")

ESTIMATOR = np.median


def occlusion_model_plots(dataframe, palette="Blues", legend_title=None, filename=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    sns.boxplot(x="sequence", y="diff_t", hue="model", data=dataframe, palette=palette, showfliers=False)
    #g = sns.pointplot(x="sequence", y="diff_t", hue="model", data=dataframe, scale=0.75, palette=palette, dodge=0.64,
    #                  estimator=ESTIMATOR, ci=None)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set_ylabel("Translation error (mm)")
    ax.set_xlabel("Occlusion %")
    ax.set_ylim([0, 75])
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
    sns.boxplot(x="sequence", y="diff_r", hue="model", data=dataframe, palette=palette, showfliers=False)
    #g = sns.pointplot(x="sequence", y="diff_r", hue="model", data=dataframe, scale=0.75, palette=palette, estimator=ESTIMATOR, ci=68)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set_ylabel("Rotation error (deg)")
    ax.set_xlabel("Occlusion %")
    ax.set_ylim([0, 60])
    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")
    plt.show()


def occlusion_object_plots(dataframe, model):
    dataframe = dataframe[dataframe['model'].str.contains(model) == True]

    ax = plt.subplot("121")
    sns.pointplot(x="sequence", y="diff_t", hue="object", data=dataframe, linestyles="--", scale=0.5, estimator=ESTIMATOR, ci=68)
    sns.pointplot(x="sequence", y="diff_t", data=dataframe, color="black", label="total", estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 60])

    ax = plt.subplot("122")
    sns.pointplot(x="sequence", y="diff_r", hue="object", data=dataframe, linestyles="--", scale=0.5, estimator=ESTIMATOR, ci=68)
    sns.pointplot(x="sequence", y="diff_r", data=dataframe, color="black", label="total", estimator=ESTIMATOR, ci=68)
    ax.set_ylim([0, 35])
    plt.suptitle(model)
    plt.show()


def clean_dataframe(df):
    df = df[df['sequence'].str.contains("occlusion")]
    df = replace_occlusion_sequence_names(df)
    df = df[df.diff_r != 0.0]
    df = df[df['object'].str.contains("turtle") == False]
    #df = df[df['object'].str.contains("skull") == False]
    return df


if __name__ == '__main__':
    root_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/single_result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/single_result_dataframe"

    occlusion_translation = pd.read_csv(os.path.join(root_path, "single_translation.csv"))

    occlusion_rotation = pd.read_csv(os.path.join(root_path, "single_rotation.csv"))
    occlusion_bb = pd.read_csv(os.path.join(root_path, "single_bb.csv"))
    occlusion_resolution = pd.read_csv(os.path.join(root_path, "single_resolution.csv"))

    occlusion_resolution = occlusion_resolution[occlusion_resolution['model'].str.contains("224") == False]

    occlusion_translation = clean_dataframe(occlusion_translation)
    occlusion_rotation = clean_dataframe(occlusion_rotation)
    occlusion_bb = clean_dataframe(occlusion_bb)
    occlusion_resolution = clean_dataframe(occlusion_resolution)


    occlusion_translation = occlusion_translation.replace("t1r20", "10")
    occlusion_translation = occlusion_translation.replace("t2r20", "20")
    occlusion_translation = occlusion_translation.replace("t3r20", "30")
    occlusion_translation = occlusion_translation.replace("t4r20", "40")
    occlusion_translation = occlusion_translation.replace("t5r20", "50")
    occlusion_model_plots(occlusion_translation, legend_title="Translation Range (mm)",
                          filename=os.path.join(output_path, "single_occlusion_deltaT"))


    occlusion_rotation = occlusion_rotation.replace("t3r15", "15")
    occlusion_rotation = occlusion_rotation.replace("t3r20", "20")
    occlusion_rotation = occlusion_rotation.replace("t3r25", "25")
    occlusion_rotation = occlusion_rotation.replace("t3r30", "30")
    occlusion_rotation = occlusion_rotation.replace("t3r35", "35")
    occlusion_model_plots(occlusion_rotation, legend_title="Rotation Range (degree)",
                          filename=os.path.join(output_path, "single_occlusion_deltaR"))

    sequences = ['0', '15', '30', '45', '60', '75']
    names = ["10", "20", "30", "40", "50"]

    matrice_t = np.zeros((len(sequences), len(names)))
    medians = occlusion_translation.groupby(['sequence', 'model'])['diff_t'].median().to_dict()
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_t[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_t[i, j] = -1

    matrice_r = np.zeros((len(sequences), len(names)))
    medians = occlusion_translation.groupby(['sequence', 'model'])['diff_r'].median().to_dict()
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


    occlusion_bb = occlusion_bb.replace("bb0", "0%")
    occlusion_bb = occlusion_bb.replace("t3r20", "10%")
    occlusion_bb = occlusion_bb.replace("bb25", "25%")
    occlusion_bb = occlusion_bb.replace("bbm10", "-10%")
    occlusion_bb = occlusion_bb.replace("bbm25", "-25%")
    occlusion_model_plots(occlusion_bb,
                          legend_title="Bounding box width",
                          filename=os.path.join(output_path, "single_occlusion_bb"))

    occlusion_resolution = occlusion_resolution.replace("r124", "124x124")
    occlusion_resolution = occlusion_resolution.replace("t3r20", "150x150")
    occlusion_resolution = occlusion_resolution.replace("r174", "174x174")
    occlusion_resolution = occlusion_resolution.replace("r200", "200x200")
    occlusion_resolution = occlusion_resolution.replace("r224", "224x224")
    occlusion_model_plots(occlusion_resolution, legend_title="Resolution",
                          filename=os.path.join(output_path, "single_occlusion_res"))

