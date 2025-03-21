import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
from tpcp import Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction


class CPointExtraction_Koka2022(BaseExtraction):
    """algorithm to extract C-points from ICG derivative signal using neurokit2s ecg_peaks() with the method koka2022"""


    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts C-points (max of most prominent peak) from given cleaned ICG derivative signal

        Args:
            signal_clean:
                cleaned ICG derivative signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ICG derivative signal in hz

        Returns:
            saves resulting C-point positions in points_, index is heartbeat id
        """

        # result df
        c_points = pd.DataFrame(index=heartbeats.index, columns=["c_point"])

        
        # search C-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():

            # slice signal for current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]
            heartbeat_icg_der = signal_clean.iloc[heartbeat_start:heartbeat_end]

            # calculate R-peak position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start

            c_point = nk.ecg_peaks(heartbeat_icg_der, sampling_rate_hz, method='koka2022')[1]["ECG_R_Peaks"][0]
           
            c_points['c_point'].iloc[idx] = c_point + heartbeat_start
        #print(c_points)
        self.points_ = c_points
        return self
