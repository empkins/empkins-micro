import neurokit2 as nk
import pandas as pd
from tpcp import make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction


class QPeakExtraction_NeurokitDwt(BaseExtraction):
    """algorithm to extract Q-wave peaks (= R-wave onset) from ECG signal using neurokit's ecg_delineate function with
    discrete wavelet method"""

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts Q-wave peaks from given ECG cleaned signal

        Args:
            signal_clean:
                cleaned ECG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns:
            saves resulting Q-peak locations (samples) in points_ attribute of super class, index is heartbeat id
        """

        # some neurokit functions (for example ecg_delineate()) don't work with r-peaks input as Series, so list instead
        r_peaks = list(heartbeats["r_peak_sample"])

        _, waves = nk.ecg_delineate(signal_clean, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, method="dwt",
                                    show=True, show_type="peaks")  # show can also be set to False

        q_peaks = waves["ECG_Q_Peaks"]
        q_peaks = super().match_points_heartbeats(self, points=q_peaks, heartbeats=heartbeats)

        self.points_ = q_peaks
        return self

