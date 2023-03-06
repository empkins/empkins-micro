import pandas as pd
from scipy import signal


def clean_icg_deriv(raw_signal: pd.Series, sampling_rate_hz: int) -> pd.Series:
    """function which cleans ICG dZ/dt signal using butterworth filtering

    (low cutoff: 0.5 Hz, high cutoff: 25 Hz, see Forouzanfar 2019)

    Args:
        raw_signal: pd.Series containing the raw dZ/dt ICG signal
        sampling_rate_hz: sampling rate of ICG dZ/dt signal in hz
    Returns:
        clean_signal: pd.DataFrame containing filtered signal
    """

    sos = signal.butter(N=4, Wn=[0.5, 25], btype="bandpass", output="sos", fs=sampling_rate_hz)
    clean_signal = signal.sosfiltfilt(sos, raw_signal)
    clean_signal = pd.Series(clean_signal, index=raw_signal.index)
    return clean_signal

