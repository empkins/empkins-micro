import neurokit2 as nk
import pandas as pd
from typing import Optional


def clean_ecg(raw_signal: pd.Series, sampling_rate_hz: int, method: Optional[str] = "biosppy") -> pd.Series:
    """function to clean ECG signals using neurokit's biosppy ecg_clean function

    Args:
        raw_signal: pd.Series containing the raw ECG signal
        sampling_rate_hz: sampling rate of the ECG signal in hz
        method: cleaning method (default is "biosppy"), can be any either "neurokit" or "biosppy"

    Returns:
        clean_signal: pd.DataFrame containing filtered signal while keeping index of input signal
    """

    if method == "neurokit":
        clean_signal = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate_hz, method=method)
        clean_signal = pd.Series(clean_signal, index=raw_signal.index, name="ecg")

    elif method == "biosppy":
        clean_signal = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate_hz, method=method)
        clean_signal = pd.Series(clean_signal, index=raw_signal.index, name="ecg")

    else:
        raise ValueError("Not implemented yet!")

    return clean_signal

