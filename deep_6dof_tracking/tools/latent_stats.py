import os
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import random


def load(path):
    latent_rgb = np.load(os.path.join(path, "latent_rgb.npy"))
    latent_rgbd = np.load(os.path.join(path, "latent_rgbd.npy"))
    return latent_rgb, latent_rgbd


def compute_erros_stats(error):
    l1 = np.mean(np.abs(error))
    l2 = np.sqrt(np.mean(np.power(error, 2)))
    return l1, l2


def plot_distribution(path, latent_rgb, latent_rgbd):
    embeding_path = os.path.join(path, "embedding.npy")
    if not os.path.exists(embeding_path):
        full = np.concatenate((latent_rgb, latent_rgbd))
        embedding = TSNE(n_components=2).fit_transform(full)
        np.save(embeding_path, embedding)
    else:
        embedding = np.load(embeding_path)

    data_size = len(latent_rgb)
    colors = sns.diverging_palette(220, 20, n=7)
    sns.scatterplot(x=embedding[:data_size, 0], y=embedding[:data_size, 1], color=colors[0], label="rgb",
                    edgecolor=colors[2])
    sns.scatterplot(x=embedding[data_size:, 0], y=embedding[data_size:, 1], color=colors[-1], label="rgbd",
                    edgecolor=colors[-3])

    colors = sns.color_palette("hls", 8)
    for color in colors:
        index = random.randint(0, data_size)
        sns.scatterplot(x=[embedding[index, 0], embedding[index+data_size, 0]],
                        y=[embedding[index, 1], embedding[index+data_size, 1]],
                        color=color, marker="X", s=150, edgecolor='black')


if __name__ == '__main__':
    folder_path_md = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/latent_space/moddroppp3"
    folder_path_mdpp = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/latent_space/moddroppp3"

    latent_rgb_md, latent_rgbd_md = load(folder_path_md)
    latent_rgb_mdpp, latent_rgbd_mdpp = load(folder_path_mdpp)

    l1_md, l2_md = compute_erros_stats(latent_rgb_md - latent_rgbd_md)
    l1_mdpp, l2_mdpp = compute_erros_stats(latent_rgb_mdpp - latent_rgbd_mdpp)

    #plt.subplot(1, 2, 1)
    plot_distribution(folder_path_md, latent_rgb_md, latent_rgbd_md)
    plt.tight_layout()
    plt.savefig("/home/mathieu/Downloads/chinese_virus/out.pdf")
    exit()
    plt.title("Moddrop")
    plt.subplot(1, 2, 2)
    plot_distribution(folder_path_mdpp, latent_rgb_mdpp, latent_rgbd_mdpp)
    plt.title("Moddrop++")

    plt.show()
