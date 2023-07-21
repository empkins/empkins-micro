from tpcp import Algorithm, Parameter, make_action_safe

import pandas as pd
import numpy as np
from neurokit2 import ecg_process
from scipy.signal import gaussian

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