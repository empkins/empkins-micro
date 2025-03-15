import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__all__ = ["emotion_plot"]


def emotion_plot(emotion_data: pd.DataFrame, **kwargs):
    fig, ax = _plot_get_fig_ax(**kwargs)

    emotion_data = emotion_data.reset_index()
    if "time_sec" not in emotion_data.columns:
        emotion_data = _get_s(emotion_data)

    emotion_data = emotion_data.set_index("time_sec")
    # emotion_data.index = pd.to_datetime(datetime.datetime.now()).normalize() + emotion_data.index

    if "condition" in emotion_data.columns:
        condition_list = []
        for condition, data in emotion_data.groupby("condition", sort=False):
            condition_list.append(condition)
            data[["emotion_numeric"]].plot(ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        labels = condition_list
        ax.legend(handles, labels, title="Condition")
    else:
        emotion_data[["emotion_numeric"]].plot(ax=ax)

    emotion_order = kwargs.get("emotion_order", None)
    if emotion_order:
        ax.set_yticks(np.arange(0, len(emotion_order)))
        ax.set_yticklabels(emotion_order)

    ax.set_ylabel("Emotion")
    ax.set_xlabel("Time")
    # print(emotion_data.index)
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.tight_layout()
    return fig, ax


def _get_s(data: pd.DataFrame):
    if "condition" in data.columns:
        data = data.groupby("condition").apply(lambda df: _get_s_per_group(df))
    else:
        data = _get_s_per_group(data)
    return data


def _get_s_per_group(data: pd.DataFrame):
    data["time_sec"] = data["time"] - data["time"].iloc[0]
    # data["time_sec"] = bp.utils.time.timedelta_to_time(data["time_sec"])
    data = data.set_index("time", append=True)
    return data


def _plot_get_fig_ax(**kwargs):
    ax: plt.Axes = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize"))
    else:
        fig = ax.get_figure()
    return fig, ax
