import warnings
from typing import Union, Optional, Dict

import numpy as np
import pandas as pd
from tpcp import Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction
from empkins_micro.feature_extraction.pep.algorithms.icg.extraction_c_point_scipy_findpeaks import \
    CPointExtraction_ScipyFindPeaks


class BPointExtraction_ThirdDeriv(BaseExtraction):
    """algorithm to extract B-point from cleaned ICG dZ/dt signal using the third derivative (see Arbol 2017)"""

    # input parameters
    window_b_detection_ms: Parameter[Union[str, int]]  # either 'R' or integer defining window length in ms
    save_icg_derivatives: Parameter[bool]

    # results
    icg_derivatives_: Dict[int, pd.DataFrame]

    def __init__(
            self,
            window_b_detection_ms: Optional[Union[str, int]] = 150,
            save_icg_derivatives: Optional[bool] = False
    ):
        """initialize new BPointExtraction_ThirdDeriv algorithm instance

        Args:
            window_b_detection_ms : str, int
                defines the window in which the algorithm searches for the B-point,
                'R' -> search B-point in the region between R-peak and C-point,
                int -> search B-point in the region between xx ms before C-point and C-point
                (150 ms -> see Arbol 2017, procedure for visual detection; Lababidi 1970)
                (300 ms -> see Arbol 2017, 3rd derivative-based algorithm)
            save_icg_derivatives : bool
                when True 2nd and 3rd derivative of ICG signal are saved in icg_derivatives_, otherwise not
        """

        self.window_b_detection_ms = window_b_detection_ms
        self.save_icg_derivatives = save_icg_derivatives

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts B-points from given cleaned ICG derivative signal

        Args:
            signal_clean :
                cleaned ICG derivative signal
            heartbeats :
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz :
                sampling rate of ICG derivative signal in hz

        Returns:
            saves resulting B-point positions in points_ attribute of super class, index is heartbeat id, and saves ICG
            derivative signal and 2nd and 3rd derivatives of each heartbeat as pd.DataFrame in icg_derivatives_
            dictionary with heartbeat id as key
        """

        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point"])
        icg_derivatives = {}
        if self.save_icg_derivatives:
            icg_derivatives = {k: pd.DataFrame(columns=["icg_der", "icg_2nd_der", "icg_3rd_der"]) for k in
                               heartbeats.index}

        # get C-points of given ICG derivative signal
        c_points = self.get_c_points(signal_clean, heartbeats, sampling_rate_hz)

        # search B-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():

            # slice signal for current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]
            heartbeat_icg_der = signal_clean.iloc[heartbeat_start:heartbeat_end]

            # calculate derivatives for this heartbeat
            heartbeat_icg_2nd_der = np.gradient(heartbeat_icg_der)
            heartbeat_icg_3rd_der = np.gradient(heartbeat_icg_2nd_der)

            if self.save_icg_derivatives:
                icg_derivatives[idx]["icg_der"] = heartbeat_icg_der
                icg_derivatives[idx]["icg_2nd_der"] = heartbeat_icg_2nd_der
                icg_derivatives[idx]["icg_3rd_der"] = heartbeat_icg_3rd_der
                self.icg_derivatives_ = icg_derivatives

            # calculate R-peak and C-point position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start
            heartbeat_c_point = c_points["c_point"].loc[idx].item() - heartbeat_start

            # set window end to C-point position and set window start according to specified method
            window_end = heartbeat_c_point
            if self.window_b_detection_ms is "R":
                window_start = heartbeat_r_peak
            elif isinstance(self.window_b_detection_ms, int):
                window_length_samples = int((self.window_b_detection_ms / 1000) * sampling_rate_hz)
                window_start = heartbeat_c_point - window_length_samples
            else:
                window_start = np.NaN
                window_end = np.NaN
                warnings.warn("That should never happen!")

            # find max in B window and calculate B-point relative to signal start
            heartbeat_b_window = heartbeat_icg_3rd_der[window_start:window_end]
            b_window_max = np.argmax(heartbeat_b_window)
            b_point = b_window_max + window_start + heartbeat_start

            b_points.at[idx, "b_point"] = b_point

        self.points_ = b_points
        return self

    @staticmethod
    def get_c_points(signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):

        c_extractor = CPointExtraction_ScipyFindPeaks(window_c_correction=3, save_candidates=False)
        c_extractor.extract(signal_clean=signal_clean, heartbeats=heartbeats, sampling_rate_hz=sampling_rate_hz)
        c_points = c_extractor.points_

        return c_points
