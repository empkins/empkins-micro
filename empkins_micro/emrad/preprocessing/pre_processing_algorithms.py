from empkins_io.datasets.d03.micro_gapvii._dataset import MicroBaseDataset

from tpcp import Algorithm, Parameter, make_action_safe

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, decimate, gaussian
from neurokit2 import ecg_process

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
        b, a = butter(self.high_pass_filter_order, normal_cutoff, btype = "high", analog = False)
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
        b, a = butter(self.low_pass_filter_order, normal_cutoff, btype = "low", analog = False)
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
        b, a = butter(self.band_pass_filter_order, [low, high], btype='band', analog = False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)

        return self
    
class ComputeEnvelopeSignal(Algorithm):
    _action_methods = "compute"

    # Input Parameters
    average_length: Parameter[int]

    # Results
    envelope_signal_: pd.Series

    def __init__(
        self,
        average_length: int = 100
    ):
        self.average_length = average_length

    @make_action_safe
    def compute(self, radar_data: pd.Series):
        """Compute the envelope of an underlying signal using same-length convolution with a normalized impulse train 

        Args:
            radar_data (pd.Series): rad (magnitude/power of complex radar signal usually already bandpass filtered to heart sound range (16-80Hz))

        Returns:
            pd.Series: envelope signal
        """
        self.envelope_signal_ = np.convolve(np.abs(hilbert(radar_data)).flatten(), np.ones(self.average_length)/self.average_length, mode='same')

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
    
class ComputeEcgPeakGaussians(Algorithm):
    _action_methods = "compute"

    # Input Parameters
    method: Parameter[str]
    gaussian_length: Parameter[int]
    gaussian_std: Parameter[float]

    # Results
    peak_gaussians_: pd.Series

    def __init__(
        self,
        gaussian_length: int = 400,
        gaussian_std: float = 6
    ):
        self.gaussian_length = gaussian_length
        self.gaussian_std = gaussian_std

    @make_action_safe
    def compute(self, ecg_signal: pd.Series, sampling_freq: float):
        """Compute the target signal with gaussians positioned at the ecg's R-peaks

        Args:
            ecg_signal (pd.Series): ECG-signal being ground-truth, containing the peaks.
            sampling_freq (float): Sampling frequency of the ECG signal.

        Returns:
            pd.Series: Signal with Gaussians located at the R-peaks of the ECG signal.
        """
        signal, info = ecg_process(ecg_signal, sampling_freq)
  
        self.peak_gaussians_ = np.convolve(signal["ECG_R_Peaks"], gaussian(self.gaussian_length, self.gaussian_std), mode='same')

        return self