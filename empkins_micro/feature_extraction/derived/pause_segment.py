import pandas as pd


def pause_features(data):
    features = {
        "aco_pausetime_mean": [data["aco_pausetime"].sum()],
        "aco_totaltime_mean": [data["aco_totaltime"].sum()],
        "aco_numpauses_mean": [data["aco_numpauses"].sum()],
        "aco_pausefrac_mean": [data["aco_pausetime"].sum() / data["aco_totaltime"].sum()]
    }
    return pd.DataFrame.from_dict(features)
