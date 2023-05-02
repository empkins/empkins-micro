import pandas as pd
from numpy.linalg import norm
from scipy import signal
import numpy as np


def scg_norm(xyz_acc_signals: pd.DataFrame) -> pd.Series:
    """function to clean SCG signal

    Args:
        xyz_acc_signals:
            pd.DataFrame containing 3 columns for x, y, z SCG components (acc), should be cleaned previously

    Returns:
        signal_norm: pd.Series containing the L2-norm of the 3 components
    """

    # check if raw_signal contains 3 columns
    if xyz_acc_signals.shape[1] != 3:
        raise ValueError(
            f"The input signal must contain 3 columns (instead of {xyz_acc_signals.shape[1]}) for x, y, z components!")

    calculated_norm = norm(xyz_acc_signals, axis=1)
    signal_norm = pd.Series(calculated_norm, index=xyz_acc_signals.index, name="acc_norm")
    return signal_norm


def scg_highpass(scg_signal: pd.Series, sampling_rate_hz: float) -> pd.Series:
    """performs high pass filtering of the SCG signals norm

    Args:
        scg_signal:
            pd.Series containing L2-norm of the three acc channels of the SCG signal
        sampling_rate_hz:
            sampling rate of SCG signal in hz

    Returns:
        filtered_signal: pd.Series containing the highpass filtered signal norm
    """

    sos = signal.butter(N=4, Wn=20, btype="highpass", output="sos", fs=sampling_rate_hz)
    filtered_signal = signal.sosfiltfilt(sos, scg_signal)
    filtered_signal = pd.Series(filtered_signal, index=scg_signal.index, name=f"{scg_signal.name}_hp")

    return filtered_signal


def scg_lowpass(scg_signal: pd.Series, sampling_rate_hz: float) -> pd.Series:
    """performs low pass filtering of the SCG signals norm

    Args:
        scg_signal:
            pd.Series containing L2-norm of the three acc channels of the SCG signal
        sampling_rate_hz:
            sampling rate of SCG signal in hz

    Returns:
        filtered_signal: pd.Series containing the lowpass filtered signal norm
    """

    sos = signal.butter(N=4, Wn=20, btype="lowpass", output="sos", fs=sampling_rate_hz)
    filtered_signal = signal.sosfiltfilt(sos, scg_signal)
    filtered_signal = pd.Series(filtered_signal, index=scg_signal.index, name=f"{scg_signal.name}_lp")

    return filtered_signal


def scg_envelope(scg_signal: pd.Series) -> pd.Series:

    envelope = signal.hilbert(scg_signal)
    amplitude_envelope = np.abs(envelope)
    envelope = pd.Series(amplitude_envelope, index=scg_signal.index, name="envelope")

    return envelope
