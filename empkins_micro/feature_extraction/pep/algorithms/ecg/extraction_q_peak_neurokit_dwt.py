import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
from tpcp import make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction


class QPeakExtraction_NeurokitDwt(BaseExtraction):
    """algorithm to extract Q-wave peaks (= R-wave onset) from ECG signal using neurokit ecg_delineate function with
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
            saves resulting Q-peak locations (samples) in points_ attribute of super class (in the row of the heartbeat
            to which the respective Q-peak corresponds), index is heartbeat id,
            NaN when no Q-peak could be detected in that heartbeat
        """

        # result df
        q_peaks = pd.DataFrame(index=heartbeats.index, columns=["q_peak"])

        # some neurokit functions (for example ecg_delineate()) don't work with r-peaks input as Series, so list instead
        r_peaks = list(heartbeats["r_peak_sample"])

        _, waves = nk.ecg_delineate(signal_clean, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, method="dwt",
                                    show=False, show_type="peaks")  # show can also be set to False

        extracted_q_peaks = waves["ECG_Q_Peaks"]

        # find heartbeat to which Q-peak belongs and save Q-peak position in corresponding row
        for idx, q in enumerate(extracted_q_peaks):

            # for some heartbeats, no Q can be detected, will be NaN in resulting df
            if np.isnan(q):
                warnings.warn(f"FYI: No Q-peak found (in heartbeat #{idx})!", UserWarning)
            else:
                heartbeat_idx = heartbeats.loc[(heartbeats["start_sample"] < q) & (q < heartbeats["end_sample"])].index
                q_peaks["q_peak"].loc[heartbeat_idx] = q

                if heartbeats["r_peak_sample"].loc[heartbeat_idx].item() < q:
                    warnings.warn(
                        f"Detected Q-peak occurs after R (wrong order) in heartbeat {heartbeat_idx}. Double check?!",
                        UserWarning)

        self.points_ = q_peaks
        return self
