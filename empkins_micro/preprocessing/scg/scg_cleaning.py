from typing import Union, Optional

import pandas as pd
from numpy.linalg import norm
from scipy import signal


def clean_scg(raw_signal: Union[pd.Series, pd.DataFrame], sampling_rate_hz: int,
              filter_type: Optional[str] = "butterworth") -> Union[pd.Series, pd.DataFrame]:
    """function to clean SCG signal

    Args:
        raw_signal: pd.Series containing a single acc channel, or pd.DataFrame containing several acc channels
        sampling_rate_hz: sampling rate of the SCG signal in hz
        filter_type: description of filter that should be used for cleaning the signal (default butterworth)

    Returns:
        clean_signal: either pd.Series or pd.DataFrame, depending on the input
    """

    # TODO: add other methods for cleaning ?

    if filter_type == "butterworth":

        sos = signal.butter(N=4, Wn=[5, 40], btype="bandpass", output="sos", fs=sampling_rate_hz)

        if isinstance(raw_signal, pd.Series):
            filtered = signal.sosfiltfilt(sos, raw_signal)
            clean_signal = pd.Series(filtered, index=raw_signal.index, name=raw_signal.name)
            return clean_signal

        elif isinstance(raw_signal, pd.DataFrame):
            clean_signal = raw_signal.apply(lambda x: signal.sosfiltfilt(sos, x), axis=0)
            return clean_signal

        else:
            raise ValueError("Input (raw_signal) needs to be either pd.Series or pd.DataFrame")

    else:
        raise ValueError("Not implemented yet!")


def scg_norm(xyz_acc_signals: pd.DataFrame) -> pd.Series:
    """function to clean SCG signal

    Args:
        xyz_acc_signals: pd.DataFrame containing 3 columns for x, y, z SCG components (acc), should be cleaned previously

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

