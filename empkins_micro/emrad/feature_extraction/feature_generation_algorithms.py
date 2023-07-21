from tpcp import Algorithm, Parameter, make_action_safe

import numpy as np
import pandas as pd
from scipy.signal import hilbert

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