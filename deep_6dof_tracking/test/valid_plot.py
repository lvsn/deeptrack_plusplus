import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    val_path = "validation_data.csv"

    labels = ["load time", "batch time", "loss", "mse", "tx", "ty", "tz", "rx", "ry", "rz"]
    val_data = pd.read_csv(val_path, header=None)
    val_column_tags = val_data.columns.values

    fig, ax = plt.subplots(3, 1)
    # Timing plot
    val_data.plot(x=np.arange(len(val_data)), y=val_column_tags[:2], ax=ax[0])
    time_labels = ["valid load time", "valid batch time"]
    ax[0].legend(time_labels)

    # loss plot
    val_data.plot(x=np.arange(len(val_data)), y=val_column_tags[2:4], ax=ax[1])
    loss_labels = ["valid loss", "valid mse"]
    ax[1].legend(loss_labels)

    # individual plot
    val_data.plot(x=np.arange(len(val_data)), y=val_column_tags[4:], ax=ax[2])
    loss_labels = ["valid tx", "valid ty", "valid tz", "valid rx", "valid ry", "valid rz"]
    ax[2].legend(loss_labels)

    plt.show()
