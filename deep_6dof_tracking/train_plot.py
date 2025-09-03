from pytorch_toolbox.logger import Logger
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse



def load_loggers(path):
    train_log = Logger()
    valid_log = Logger()
    train_log.load(os.path.join(path, "training_data.log"))
    valid_log.load(os.path.join(path, "validation_data.log"))
    return train_log, valid_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepTrack')
    parser.add_argument('-o', '--output', help="Output path", action="store", default="")
    parser.add_argument('-r', '--root', help="log files", action="store", default="")

    arguments = parser.parse_args()

    root_path = arguments.root
    output_path = arguments.output

    keys = ["DOF", "tx", "ty", "tz", "rx", "ry", "rz"]

    train_log1, valid_log1 = load_loggers(root_path)
    train_log2, valid_log2 = load_loggers(root_path)

    plt.figure(figsize=(20, 20), dpi=80)
    plt.title('clock')

    for i, key in enumerate(keys):
        plt.subplot(len(keys), 2, i * 2 + 1)
        #ax = sns.lineplot(x=np.arange(len(train_log1[key])), y=train_log1[key], color="blue", label="Train {}".format("name1"), marker="o")
        ax = sns.lineplot(x=np.arange(len(train_log2[key])), y=train_log2[key], color="red", label="Train {}".format("name2"), marker="o")
        ax.legend()
        plt.title(key)

        plt.subplot(len(keys), 2, i * 2 + 2)
        ax = sns.lineplot(x=np.arange(len(valid_log1[key])), y=valid_log1[key], color="blue", label="Valid {}".format("name1"), marker="o")
        #ax = sns.lineplot(x=np.arange(len(valid_log2[key])), y=valid_log2[key], color="red", label="Valid {}".format("name2"), marker="o")
        ax.legend()
        plt.title(key)
    # plt.savefig('train_clock_pml.png')
    plt.savefig(output_path)
        