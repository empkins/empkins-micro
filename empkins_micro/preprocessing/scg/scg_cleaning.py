from typing import Union, Optional

import pandas as pd
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
            filtered_signal = signal.sosfiltfilt(sos, raw_signal)
            clean_signal = pd.Series(filtered_signal, index=raw_signal.index, name=raw_signal.name)
            return clean_signal

        elif isinstance(raw_signal, pd.DataFrame):
            #clean_signal = pd.DataFrame(columns=raw_signal.columns, index=raw_signal.index)
            filtered_signals = raw_signal.apply(lambda x: signal.sosfiltfilt(sos, x), axis=0)
            return filtered_signals

        else:
            raise ValueError("Input (raw_signal) needs to be either pd.Series or pd.DataFrame")

    else:
        raise ValueError("Not implemented yet!")
