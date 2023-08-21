from tpcp import Algorithm, make_action_safe
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks

class PairwiseHeartRate(Algorithm):
    """Algorithm object applying the scipy.signal find_peaks() function to find R-Peaks in the predicted R-Peak signal from the biLSTMPipelineNo1.
    Afterwards it calculates the instantaneous heart rate between two found peaks and interpolates a function between the calculated heart rates.
    The result is this interpolated continous function sampled with 1Hz.

    Attributes:
        
    """

    _action_methods = "compute"

    # INPUT PARAMETERS
    max_heart_rate: int
    sampling_rate: float

    # Results
    heart_rate_: np.ndarray

    def __init__(
        self,
        max_heart_rate: int = 180,
        sampling_rate: float = 100
    ):
        self.max_heart_rate = max_heart_rate
        self.sampling_rate = sampling_rate

        
    @make_action_safe
    def compute(self, input_data: np.ndarray):
        """Accepts a R peak prediciton signal and computes the instantaneous heart rates sampled with 1 Hz.

        Args:
            input_data (np.ndarray): R Peak prediction signal

        Returns:
            self
        """

        input_data = np.squeeze(input_data)

        minimal_distance_between_peaks =  int(1 / (self.max_heart_rate / 60) * self.sampling_rate)
        
        peaks, _ = find_peaks(input_data, distance=minimal_distance_between_peaks, height=0.2)

        sample_diffs_between_peaks = np.diff(peaks)

        #convert sample diff to time diff
        time_diffs_between_peaks = sample_diffs_between_peaks / self.sampling_rate

        #convert time diff to heart rate
        instantaneous_heart_rates = 60 / time_diffs_between_peaks 

        #heart rates in the middle of two peaks
        heart_rate_signal_x = peaks[:-1] + (sample_diffs_between_peaks / 2).astype(int)

        #get interpolation function (values outside of peaks will be extrapolated)
        f = interpolate.interp1d(heart_rate_signal_x, instantaneous_heart_rates, fill_value='extrapolate')

        sample_x = np.arange(0, input_data.shape[0], self.sampling_rate)

        #compute heart rate at 1Hz sample points
        self.heart_rate_ = f(sample_x)
        
        return self
    