import numpy as np


def mean(data):
    df = data.astype(float).copy()
    df = df.dropna().reset_index(drop=True)
    val = df.mean(axis=0, skipna=True)
    return val


def std(data):
    df = data.astype(float).copy()
    df = df.dropna().reset_index(drop=True)
    val = df.std(axis=0, skipna=True)
    return val


def range(data):
    df = data.astype(float).copy()
    df = df.dropna().reset_index(drop=True)
    val = max(df) - min(df)
    return val


def pct(data):
    df = data.astype(float).copy()
    df = df.dropna().reset_index(drop=True)
    val = len(df[df > 0.0]) / len(df)
    return val


def count(data):
    val = sum(data) * 60.0 / data.index[-1]
    return val


def _calc_blinks_diff(data):
    time = data.index.to_numpy()
    indices = np.where(data == 1)[0]
    time_blinks = time[indices]
    time_blinks_diff = np.diff(time_blinks)
    return time_blinks_diff


def dur_mean(data):
    time_blinks_diff = _calc_blinks_diff(data)
    val = np.mean(time_blinks_diff)
    return val


def dur_std(data):
    time_blinks_diff = _calc_blinks_diff(data)
    val = np.std(time_blinks_diff)
    return val


def mean_weighted(data, weights):
    indices = data.isna()
    df = data.astype(float).copy()
    df = df.dropna().reset_index(drop=True)
    weights = weights[~indices]
    val = np.average(df, weights=weights)
    return val


def std_weighted(data, weights):
    indices = data.isna()
    df = data.astype(float).copy()
    df = df.dropna().reset_index(drop=True)
    weights = weights[~indices]
    val = np.sqrt(np.cov(df.to_numpy(), aweights=weights))
    return val

