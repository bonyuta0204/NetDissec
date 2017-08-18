# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess
from pathlib import Path


IOU = ["color-iou", "object-iou", "scene-iou",
       "texture-iou", "part-iou", "material-iou"]


def plot_analysis(data_csv, ax=None, name=" ", error_bar=True):
    """
    plot bar graph for given result

    parameter:
        data_csv: list
            list of path for result(.csv)
        ax: matplotlib figure
             ax to use
        name: str
            name for title of ax
        error_bar:bool
            if  True, show error bar
    """
    if ax is None:
        ax = plt.gca()
    else:
        ax = ax
    num_bars = len(IOU)
    tick = np.arange(1, num_bars + 1, 1, dtype=np.float32)
    org_tick = [tic for tic in tick]
    width = 0.7 / len(data_csv)

    # plot bars for each data
    for csv in data_csv:
        result = pd.read_csv(csv)
        result = result[IOU]
        stats = result.describe().T
        mean = np.array(stats["mean"])
        if error_bar:
            std = np.array(stats["std"])
            ax.bar(tick, mean, width=width,
                   label=csv.name, yerr=std, capsize=5.0)
        else:
            ax.bar(tick, mean, width=width, label=csv.name)
        tick += width

    # set ticks
    label_ticks = (tick + org_tick - width) / 2
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(IOU)
    ax.set_ylim(bottom=0)

    # other configuration
    ax.legend()
    ax.set_title("Interpretability")
    ax.set_ylabel("Interpretability")
    return ax

if __name__ == "__main__":
    cd = Path.cwd()
    csvs = cd.rglob("*.csv")
    csvs = [csv for csv in csvs]
    ax = plot_analysis(csvs)

    plt.show()
