import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
font_size = 22
sns.set_style("whitegrid", {"xtick.major.size": 4, "ytick.major.size": 4})
ESTIMATOR = np.median


def fix_paper(df, legend_title, palette="Blues", filename=None):

    df = df[df['sequence'].str.contains("fix")]
    df.loc[df['sequence'].str.contains("near"), 'sequence'] = "near"
    df.loc[df['sequence'].str.contains("far"), 'sequence'] = "far"
    df.loc[df['sequence'].str.contains("occluded"), 'sequence'] = "occluded"
    order = ["near", "far", "occluded"]
    #delta_t = 0.055
    #df.speed_t /= delta_t
    #df.speed_r /= delta_t

    plt.figure(figsize=(12, 8))
    ax = plt.subplot("111")
    sns.boxplot(ax=ax, x="sequence", y="speed_t", data=df, palette=palette, hue="model", order=order, showfliers=False)
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    ax.set(xlabel="Sequence Type", ylabel='Translation speed (mm/frame)')
    plt.ylim([0, 3.2])

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
    sns.boxplot(ax=ax, x="sequence", y="speed_r", data=df, palette=palette, hue="model", order=order,
                  showfliers=False)
    ax.set(xlabel="Sequence Type", ylabel='Rotation speed (degree/frame)')
    leg = ax.legend(loc="upper left", title=legend_title, prop={'size': 14}, frameon=True)
    leg.get_frame().set_alpha(1)
    plt.ylim([0, 3.2])

    # change font sizes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.setp(ax.get_legend().get_title(), fontsize=font_size)  # for legend title

    #plt.suptitle("Stability")
    plt.tight_layout()
    if filename:
        plt.savefig(filename + "_R.pdf")
    plt.show()


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/eccv_backup/single_result_dataframe"

    fix_translation = pd.read_csv(os.path.join(root_path, "single_translation.csv"))
    fix_rotation = pd.read_csv(os.path.join(root_path, "single_rotation.csv"))
    fix_bb = pd.read_csv(os.path.join(root_path, "single_bb.csv"))
    fix_resolution = pd.read_csv(os.path.join(root_path, "single_resolution.csv"))

    fix_translation = fix_translation[fix_translation['frame_id'] > 15]
    fix_rotation = fix_rotation[fix_rotation['frame_id'] > 15]
    fix_bb = fix_bb[fix_bb['frame_id'] > 15]
    fix_resolution = fix_resolution[fix_resolution['frame_id'] > 15]

    fix_translation = fix_translation[fix_translation['object'].str.contains("turtle") == False]
    fix_rotation = fix_rotation[fix_rotation['object'].str.contains("turtle") == False]
    fix_bb = fix_bb[fix_bb['object'].str.contains("turtle") == False]
    fix_resolution = fix_resolution[fix_resolution['object'].str.contains("turtle") == False]
    fix_resolution = fix_resolution[fix_resolution['model'].str.contains("r224") == False]

    #fix_translation = fix_translation[fix_translation['object'].str.contains(object) == False]
    #fix_rotation = fix_rotation[fix_rotation['object'].str.contains(object) == False]

    fix_translation = fix_translation.replace("t1r20", "10")
    fix_translation = fix_translation.replace("t2r20", "20")
    fix_translation = fix_translation.replace("t3r20", "30")
    fix_translation = fix_translation.replace("t4r20", "40")
    fix_translation = fix_translation.replace("t5r20", "50")
    fix_paper(fix_translation, "Translation range (mm)",
              filename=os.path.join(output_path, "single_stability_deltaT"))


    fix_rotation = fix_rotation.replace("t3r15", "15")
    fix_rotation = fix_rotation.replace("t3r20", "20")
    fix_rotation = fix_rotation.replace("t3r25", "25")
    fix_rotation = fix_rotation.replace("t3r30", "30")
    fix_rotation = fix_rotation.replace("t3r35", "35")
    fix_paper(fix_rotation, "Rotation range (degree)",
              filename=os.path.join(output_path, "single_stability_deltaR"))

    fix_rotation.loc[fix_rotation['sequence'].str.contains("near"), 'sequence'] = "near"
    fix_rotation.loc[fix_rotation['sequence'].str.contains("far"), 'sequence'] = "far"
    fix_rotation.loc[fix_rotation['sequence'].str.contains("occluded"), 'sequence'] = "occluded"

    sequences = ["near", "far", "occluded"]
    names = ["15", "20", "25", "30", "35"]
    matrice_r = np.zeros((len(sequences), len(names)))
    medians = fix_rotation.groupby(['sequence', 'model'])['speed_r'].median().to_dict()
    print(medians)

    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_r[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_r[i, j] = -1

    matrice_t = np.zeros((len(sequences), len(names)))
    medians = fix_rotation.groupby(['sequence', 'model'])['speed_t'].median().to_dict()
    for i, sequence in enumerate(sequences):
        for j, model in enumerate(names):
            try:
                matrice_t[i, j] = medians[(sequence, model)]
            except KeyError:
                matrice_r[i, j] = -1

    string_r = ""
    string_t = ""
    total_string = ""
    for j, model in enumerate(names):
        string_t += names[j]
        for i, sequence in enumerate(sequences):
            string_t += " & {:.2f}".format(matrice_t[i, j])
            string_r += " & {:.2f}".format(matrice_r[i, j])
        total_string += (string_t + string_r) + " \\\\ \n"
        string_t = ""
        string_r = ""
    print(total_string)

    fix_bb = fix_bb.replace("bb0", "0%")
    fix_bb = fix_bb.replace("t3r20", "10%")
    fix_bb = fix_bb.replace("bb25", "25%")
    fix_bb = fix_bb.replace("bbm10", "-10%")
    fix_bb = fix_bb.replace("bbm25", "-25%")
    fix_paper(fix_bb, "Bounding box width",
              filename=os.path.join(output_path, "single_stability_bb"))

    fix_resolution = fix_resolution.replace("r124", "124x124")
    fix_resolution = fix_resolution.replace("t3r20", "150x150")
    fix_resolution = fix_resolution.replace("r174", "174x174")
    fix_resolution = fix_resolution.replace("r200", "200x200")
    fix_resolution = fix_resolution.replace("r224", "224x224")
    fix_paper(fix_resolution, "Resolution",
              filename=os.path.join(output_path, "single_stability_res"))

