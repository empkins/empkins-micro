import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import argrelmin

from tpcp import Algorithm, Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction

import warnings


class BPointExtractionDebski(BaseExtraction):
    """algorithm to extract B-point based on the reversal (local minimum) of dZ^^2/dt^^2 before the C point"""

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts B-points from given ICG cleaned signal

        Args:
            signal_clean:
                cleaned ICG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            c_points:
                pd.DataFrame containing one row per segmented C-point, each row contains location
                (in samples from beginning of signal) of that C-point or NaN if the location of that C-point
                is not correct
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns:
            saves resulting B-point locations (samples) in points_ attribute of super class,
            index is C-point (/heartbeat) id
        """
        # Create the b_point Dataframe with the index of the heartbeat_list
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point"])

        # get the r_peak locations from the heartbeat_list and check if any entries contain NaN
        r_peaks = heartbeats['r_peak_sample']
        check_r_peaks = np.isnan(r_peaks.values)

        # get the c_point locations and check if any entries contain NaN
        c_points = c_points['c_point']
        check_c_points = np.isnan(c_points.values.astype(float))

        # Compute the second derivative of the ICG-signal
        icg_2nd_der = np.gradient(signal_clean)

        # go trough each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_r_peaks[idx] | check_c_points[idx]:
                b_points['b_point'].iloc[idx] = np.NaN
                warnings.warn(f"Either the r_peak or the c_point contains NaN at position{idx}!")
                continue
            else:
                # set the borders of the interval between the R-Peak and the C-Point
                start_r_c = r_peaks[idx]
                end_r_c = c_points[idx]

            # Select the specific interval in the second derivative of the ICG-signal
            icg_search_window = icg_2nd_der[start_r_c:end_r_c]

            # Compute the local minima in this interval
            icg_min = argrelmin(icg_search_window)

            # Compute the distance between the C-point and the minima of the interval and select the entry with
            # the minimal distance as B-point
            if len(icg_min[0]) >= 1:
                distance = end_r_c - icg_min
                b_point_idx = distance.argmin()
                b_point = icg_min[0][b_point_idx]
                # Compute the absolute sample position of the local B-point
                b_point = b_point + start_r_c
            else:
                # If there was no minima set the B-Point to NaN
                b_point = np.NaN
                warnings.warn("Could not find a local minimum i the R-C interval!")

            # Add the found B-point to the b_points Dataframe
            b_points['b_point'].iloc[idx] = b_point

        points = b_points
        self.points_ = points
        return self
