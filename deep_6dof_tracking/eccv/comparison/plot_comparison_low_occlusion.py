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

    df = df.replace("occlusion_h_15{}".format(post), "15")
    df = df.replace("occlusion_h_30{}".format(post), "30")
    df = df.replace("occlusion_h_45{}".format(post), "45")
    df = df.replace("occlusion_h_60{}".format(post), "60")
    return df


def clean_dataframe(df):
    df = df[df['sequence'].str.contains("occlusion")]
    df = df[df.diff_r != 0]
    df = replace_occlusion_sequence_names(df)
    df = df[df['object'].str.contains("turtle") == False]
    #df = df[df['object'].str.contains("skull") == False]
    return df

if __name__ == '__main__':
    root_path = "/media/ssd/eccv/Results/result_dataframe"
    output_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/University/Redaction/Publication/2018/ECCV/ECCV2018_DeepTrack/Images/Evaluation"

    dataframe = pd.read_csv(os.path.join(root_path, "other_low.csv"))

    dataframe = clean_dataframe(dataframe)

    df = dataframe[dataframe['model'].str.contains("single")]
    #df_low = dataframe[dataframe['model'].str.contains("single_low")]

    #df_low = df_low.replace("dragon", "dragon_low")
    #df_low = df_low.replace("skull", "skull_low")
    #df_low = df_low.replace("shoe", "shoe_low")
    #df_low = df_low.replace("clock", "clock_low")

    #dataframe = df.append(df_low)

    blue_colors = sns.color_palette("Blues")
    red_colors = sns.color_palette("Reds")
    green_colors = sns.color_palette("Greens")
    yellow_colors = sns.color_palette("BrBG")
    colors = [blue_colors[-2], yellow_colors[0], blue_colors[-4], yellow_colors[1]]

    dataframe = dataframe.replace("single_low", "Low resolution object-specific")
    dataframe = dataframe.replace("general_low", "Low resolution generic")
    dataframe = dataframe.replace("single", "Ours object-specific")
    dataframe = dataframe.replace("general", "Ours generic")
    model_order = ["Ours object-specific", "Low resolution object-specific", "Ours generic", "Low resolution generic"]

    #model_order = ["shoe", "shoe_low", "clock", "clock_low", "skull", "skull_low", "dragon", "dragon_low"]

    occlusion_model_plots(dataframe, palette=colors, model_order=model_order,
                          filename=os.path.join(output_path, "comparison_occlusion_lowres"))

