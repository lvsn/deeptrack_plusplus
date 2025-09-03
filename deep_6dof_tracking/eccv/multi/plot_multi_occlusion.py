import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")

ESTIMATOR = np.median


def replace_occlusion_sequence_names(df, post=""):
    df = df.replace("occlusion_0{}".format(post), "0")
    df = df.replace("occlusion_v_15{}".format(post), "15")
    df = df.replace("occlusion_v_30{}".format(post), "30")
    df = df.replace("occlusion_v_45{}".format(post), "45")
    df = df.replace("occlusion_v_60{}".format(post), "60")

    df = df.replace("occlusion_h_15{}".format(post), "15")
    df = df.replace("occlusion_h_30{}".format(post), "30")
    df = df.replace("occlusion_h_45{}".format(post), "45")
    df = df.replace("occlusion_h_60{}".format(post), "60")
    return df


if __name__ == '__main__':
    root_path = "/media/ssd/eccv/result_dataframe"

    occlusion_translation = pd.read_csv(os.path.join(root_path, "single_translation.csv"))
    occlusion_rotation = pd.read_csv(os.path.join(root_path, "single_rotation.csv"))
    occlusion_bb = pd.read_csv(os.path.join(root_path, "single_bb.csv"))
    occlusion_resolution = pd.read_csv(os.path.join(root_path, "single_resolution.csv"))
    single_occlusion = occlusion_translation
    single_occlusion = single_occlusion[single_occlusion.diff_t != 0.0]

    dataframe_multi = pd.read_csv(os.path.join(root_path, "multi.csv"))
    dataframe_multi = dataframe_multi[dataframe_multi['sequence'].str.contains("occlusion") == True]

    dataframe_single = occlusion_translation
    dataframe_single = dataframe_single[dataframe_single['model'].str.contains("t3r20") == True]
    dataframe_single = dataframe_single.replace("t3r20", 1)
    dataframe_single.loc[:, 'is_part'] = pd.Series(True, index=dataframe_single.index)
    dataframe_single = dataframe_single[dataframe_single['sequence'].str.contains("reset15") == True]

    dataframe_single = replace_occlusion_sequence_names(dataframe_single, post="_reset15")
    dataframe_multi = replace_occlusion_sequence_names(dataframe_multi, post="")
    df = dataframe_single.append(dataframe_multi)
    df = df[df.diff_t != 0.0]
    df = df[df['object'].str.contains("turtle") == False]

    # Compare occlusion w.r.t number of object trained
    hue_order = [1, 5, 10, 20, 26]
    df_temp = df[df['is_part'] == True]
    df_squeeze = df_temp[df_temp['training'].str.contains("mse")]
    ax = plt.subplot("121")
    g = sns.pointplot(x="sequence", y="diff_t", hue="model", data=df_squeeze, hue_order=hue_order, palette="Blues",
                  estimator=ESTIMATOR, ci=68)
    g.legend_.set_title('3D models')
    ax.set_ylim([0, 60])

    ax = plt.subplot("122")
    g = sns.pointplot(x="sequence", y="diff_r", hue="model", data=df_squeeze, hue_order=hue_order, palette="Blues",
                  estimator=ESTIMATOR, ci=68)
    g.legend_.set_title('3D models')
    ax.set_ylim([0, 30])
    plt.show()

    # compare occlusion geo vs squeezenet
    hue_order = [1, 5, 10, 20, 26]
    df_temp = df[df['is_part'] == True]
    df_geo = df_temp[df_temp['training'].str.contains("projection")]
    df_squeeze = df_temp[df_temp['training'].str.contains("mse")]

    ax = plt.subplot("121")
    sns.pointplot(x="sequence", y="diff_t", hue="training", data=df_temp, palette="Greens",
                  estimator=ESTIMATOR, ci=68)
    #sns.pointplot(x="sequence", y="diff_t", hue="model", data=df_squeeze, hue_order=hue_order, palette="Blues",
    #              estimator=ESTIMATOR, ci=68)
    #new_title = 'Blue : mse\nGreen : Projection'
    #ax.legend_.set_title(new_title)
    ax.set_ylim([0, 60])

    ax = plt.subplot("122")
    sns.pointplot(x="sequence", y="diff_r", hue="training", data=df_temp, palette="Greens",
                  estimator=ESTIMATOR, ci=68)
    #sns.pointplot(x="sequence", y="diff_r", hue="model", data=df_squeeze, hue_order=hue_order, palette="Blues",
    #              estimator=ESTIMATOR, ci=68)
    #ax.legend_.set_title(new_title)
    ax.set_ylim([0, 30])
    plt.show()

    # compare occlusion geo vs squeezenet
    hue_order = [5, 10, 20, 26]
    df_temp = df[df['training'].str.contains("mse")]
    df_part = df_temp[df_temp['is_part'] == True]
    df_notpart = df_temp[df_temp['is_part'] == False]

    ax = plt.subplot("121")
    sns.pointplot(x="sequence", y="diff_t", hue="is_part", data=df_temp, palette="Blues",
                  estimator=ESTIMATOR, ci=68)
    #g = sns.pointplot(x="sequence", y="diff_t", hue="is_part", data=df_temp, hue_order=hue_order, palette="Greens",
    #                  estimator=ESTIMATOR, ci=68)
    #new_title = 'Blue : Known\nGreen : Not known'
    #g.legend_.set_title(new_title)
    #new_labels = ['Yes', 'No']
    #for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)

    ax.set_ylim([0, 60])

    ax = plt.subplot("122")
    sns.pointplot(x="sequence", y="diff_r", hue="is_part", data=df_temp, palette="Blues",
                  estimator=ESTIMATOR, ci=68)
    #g = sns.pointplot(x="sequence", y="diff_r", hue="is_part", data=df_temp, palette="Greens",
    #                  estimator=ESTIMATOR, ci=68)
    #new_title = 'Blue : Known\nGreen : Not known'
    #g.legend_.set_title(new_title)
    #new_labels = ['Yes', 'No']
    #for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)

    ax.set_ylim([0, 30])
    plt.show()
