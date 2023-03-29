import pandas as pd
from typing import Optional

from tpcp import Algorithm, Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction


class QWaveOnsetExtractionVanLien(BaseExtraction):
    """algorithm to extract Q-wave onset based on the detection of the R-peak
     and the subtraction of a fixed time interval"""

    # parameters
    time_interval: Parameter[int]

    def __init__(
            self,
            time_interval: Optional[int] = 40
    ):

        """initialize new QWaveOnsetExtractionVanLien algorithm instance

        Params:
        time_interval : int
            Specify the constant time interval which will be subtracted from the R-peak for Q-wave onset estimation
        """

        self.time_interval = time_interval

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts Q-wave onset (start of ventricular depolarization) from given ECG cleaned signal

        Args:
            signal_clean:
                cleaned ECG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns:
            saves resulting Q-wave-onset locations (samples) in points_ attribute of super class, index is heartbeat id
        """

        # convert the fixed time_interval from milliseconds into samples
        time_interval_in_samples = (self.time_interval/1000) * sampling_rate_hz  # 40 ms = 0.04 s

        # get the r_peaks from the heartbeats Dataframe
        r_peaks = heartbeats["r_peak_sample"]

        # subtract the fixed time_interval from the r_peak samples to estimate the q_wave_onset
        q_wave_onset = r_peaks - time_interval_in_samples

        points = q_wave_onset
        points = super().match_points_heartbeats(self, points=points, heartbeats=heartbeats)

        self.points_ = points
        return self
