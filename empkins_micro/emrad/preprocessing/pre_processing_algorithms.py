from tpcp import Algorithm, Parameter, make_action_safe

import pandas as pd
from scipy.signal import butter, filtfilt, decimate


class ButterHighpassFilter(Algorithm):
    _action_methods = "filter"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float]
    high_pass_filter_order: Parameter[int]

    # Results
    filtered_signal_: pd.Series

    def __init__(
            self,
            high_pass_filter_cutoff_hz: float = 0.4,
            high_pass_filter_order: int = 5
    ):
        self.high_pass_filter_cutoff_hz = high_pass_filter_cutoff_hz
        self.high_pass_filter_order = high_pass_filter_order

    @make_action_safe
    def filter(self, radar_data: pd.Series, sample_frequency_hz: float):
        """Highpass filter, filtering either I or Q of the radar data

        Args:
            radar_data (pd.Series): rad_i or rad_q
            sample_frequency_hz (float): For Radar: When aligned already with biopac is 1000Hz, raw data is 1953.125Hz.

        Returns:
            _type_: highpass-filterd signal
        """
        radar = radar_data.to_numpy().flatten()

        nyq = 0.5 * sample_frequency_hz
        normal_cutoff = self.high_pass_filter_cutoff_hz / nyq
        b, a = butter(N=self.high_pass_filter_order, Wn=normal_cutoff, btype="high", analog=False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)

        return self


class ButterLowpassFilter(Algorithm):
    _action_methods = "filter"

    # Input Parameters
    low_pass_filter_cutoff_hz: Parameter[float]
    low_pass_filter_order: Parameter[int]

    # Results
    filtered_signal_: pd.Series

    def __init__(
            self,
            low_pass_filter_cutoff_hz: float = 0.4,
            low_pass_filter_order: int = 5
    ):
        self.low_pass_filter_cutoff_hz = low_pass_filter_cutoff_hz
        self.low_pass_filter_order = low_pass_filter_order

    @make_action_safe
    def filter(self, radar_data: pd.Series, sample_frequency_hz: float):
        """Lowpass filter, filtering either I or Q of the radar data

        Args:
            radar_data (pd.Series): rad_i or rad_q
            sample_frequency_hz (float): For Radar: When aligned already with biopac is 1000Hz, raw data is 1953.125Hz.

        Returns:
            _type_: lowpass-filterd signal
        """
        radar = radar_data.to_numpy().flatten()

        nyq = 0.5 * sample_frequency_hz
        normal_cutoff = self.low_pass_filter_cutoff_hz / nyq
        b, a = butter(self.low_pass_filter_order, normal_cutoff, btype="low", analog=False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)

        return self


class ButterBandpassFilter(Algorithm):
    _action_methods = "filter"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float]
    low_pass_filter_cutoff_hz: Parameter[float]
    band_pass_filter_order: Parameter[int]

    # Results
    filtered_signal_: pd.Series

    def __init__(
            self,
            high_pass_filter_cutoff_hz: float = 80,
            low_pass_filter_cutoff_hz: float = 15,
            band_pass_filter_order: int = 5
    ):
        self.low_pass_filter_cutoff_hz = low_pass_filter_cutoff_hz
        self.high_pass_filter_cutoff_hz = high_pass_filter_cutoff_hz
        self.band_pass_filter_order = band_pass_filter_order

    @make_action_safe
    def filter(self, radar_data: pd.Series, sample_frequency_hz: float):
        """Bandpass filter, filtering the power signal of the radar

        Args:
            radar_data (pd.Series): rad (magnitude/power of complex radar signal)
            sample_frequency_hz (float): For Radar: When aligned already with biopac is 1000Hz, raw data is 1953.125Hz.

        Returns:
            _type_: bandpass-filterd signal
        """
        radar = radar_data.to_numpy().flatten()

        nyq = 0.5 * sample_frequency_hz
        low = self.low_pass_filter_cutoff_hz / nyq
        high = self.high_pass_filter_cutoff_hz / nyq
        b, a = butter(self.band_pass_filter_order, [low, high], btype='band', analog=False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)

        return self


class ComputeDecimateSignal(Algorithm):
    _action_methods = "compute"

    # Input Parameters
    downsampling_factor: Parameter[int]

    # Results
    downsampled_signal_: pd.Series

    def __init__(
            self,
            downsampling_factor: int = 20
    ):
        self.downsampling_factor = downsampling_factor

    @make_action_safe
    def compute(self, high_freq_signal: pd.Series):
        """Compute the downsampled version of a signal

        Args:
            high_freq_signal (pd.Series): High-frequency version of the signal to be downsampled.

        Returns:
            pd.Series: downsampled signal
        """
        self.downsampled_signal_ = decimate(high_freq_signal, self.downsampling_factor, axis=0)

        return self
