import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

__all__ = ["emotion_plot"]


def emotion_plot(emotion_data: pd.DataFrame, **kwargs):
    fig, ax = _plot_get_fig_ax(**kwargs)

    emotion_data = emotion_data.reset_index()
    emotion_data["time_sec"] = emotion_data["time_sec"].astype(int)
    sns.lineplot(data=emotion_data, x="time_sec", y="emotion_numeric", hue="condition", ax=ax)

    emotion_order = kwargs.get("emotion_order", None)
    if emotion_order:
        ax.set_yticks(np.arange(0, len(emotion_order)))
        ax.set_yticklabels(emotion_order)

    ax.set_ylabel("Emotion")
    if isinstance(emotion_data.index, pd.DatetimeIndex):
        ax.set_xlabel("Time")

    fig.tight_layout()
    return fig, ax


def _plot_get_fig_ax(**kwargs):
    ax: plt.Axes = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize"))
    else:
        fig = ax.get_figure()
    return fig, ax
