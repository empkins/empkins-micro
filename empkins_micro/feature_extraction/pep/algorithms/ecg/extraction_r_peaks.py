import pandas as pd
import numpy as np
import neurokit2 as nk

from tpcp import Algorithm, Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction


class RPeakExtraction(BaseExtraction):
    """algorithm to extract Q-wave onset based on the detection of the R-peak"""

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts R-peaks from given ECG cleaned signal to use it as Q-wave onset estimate

        Args:
            signal_clean:
                cleaned ECG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns:
            saves resulting R-peak locations (samples) in points_ attribute of super class, index is heartbeat id
        """

        # get the r_peaks from the heartbeats Dataframe
        r_peaks = heartbeats["r_peak_sample"]

        points = r_peaks
        points = super().match_points_heartbeats(self, points=points, heartbeats=heartbeats)

        self.points_ = points
        return self
