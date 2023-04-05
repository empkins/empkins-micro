import neurokit2 as nk
import pandas as pd
from typing import Optional


def clean_ecg(raw_signal: pd.Series, sampling_rate_hz: int, method: Optional[str] = "neurokit") -> pd.Series:
    """function to clean ECG signals using neurokit's ecg_clean function

    Args:
        raw_signal: pd.Series containing the raw ECG signal
        sampling_rate_hz: sampling rate of the ECG signal in hz
        method: cleaning method (default is "neurokit"), can be any of the options of neurokit's ecg_clean function

    Returns:
        clean_signal: pd.DataFrame containing filtered signal while keeping index of input signal
    """

    # TODO: add other cleaning methods

    if method is "neurokit":
        clean_signal = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate_hz, method=method)
        clean_signal = pd.Series(clean_signal, index=raw_signal.index, name="ecg")

    else:
        raise ValueError("Not implemented yet!")

    return clean_signal

