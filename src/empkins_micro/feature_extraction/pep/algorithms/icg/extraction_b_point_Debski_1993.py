import pandas as pd
import numpy as np
from scipy.signal import argrelmin, find_peaks
from typing import Optional

from tpcp import Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction

import warnings


class BPointExtractionDebski(BaseExtraction):
    """algorithm to extract B-point based on the reversal (local minimum) of dZ^^2/dt^^2 before the C point"""

    # input parameters
    correct_outliers: Parameter[bool]

    def __init__(
            self,
            correct_outliers: Optional[bool] = False
    ):
        """initialize new BPointExtractionDebski algorithm instance

        Parameters
        ----------
        correct_outliers : bool
            Indicates whether to perform outlier correction (True) or not (False)
        """

        self.correct_outliers = correct_outliers

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, c_points: pd.DataFrame,
                sampling_rate_hz: int):
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
        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point"])

        # get the r_peak locations from the heartbeats dataframe and search for entries containing NaN
        r_peaks = heartbeats['r_peak_sample']
        check_r_peaks = np.isnan(r_peaks.values)
        
        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        c_points = c_points['c_point']
        check_c_points = np.isnan(c_points.values.astype(float))
        
       
        # Compute the second derivative of the ICG-signal
        icg_2nd_der = np.gradient(signal_clean)

        counter = 0
        # go trough each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_r_peaks[idx] | check_c_points[idx]:
                b_points['b_point'].iloc[idx] = np.NaN
                warnings.warn(f"Either the r_peak or the c_point contains NaN at position{idx}! "
                              f"B-Point was set to NaN.")
                continue
            else:
                # set the borders of the interval between the R-Peak and the C-Point
                start_r_c = r_peaks[idx]
                end_r_c = c_points[idx]

            # Select the specific interval in the second derivative of the ICG-signal
            icg_search_window = icg_2nd_der[start_r_c:(end_r_c + 1)]

            # Compute the local minima in this interval
            #icg_min = argrelmin(icg_search_window)
            icg_min = find_peaks(-icg_search_window)[0]
            #print(icg_min)

            # Compute the distance between the C-point and the minima of the interval and select the entry with
            # the minimal distance as B-point
            if len(icg_min) >= 1:
                distance = end_r_c - icg_min
                b_point_idx = distance.argmin()
                b_point = icg_min[b_point_idx]
                # Compute the absolute sample position of the local B-point
                b_point = b_point + start_r_c
            else:
                # If there is no minima set the B-Point to NaN
                if not self.correct_outliers:
                    b_point = np.NaN
                else:
                    b_point = data['r_peak_sample']
                counter += 1

            # Add the detected B-point to the b_points Dataframe
            '''
            if not self.correct_outliers:
                if b_point < data['r_peak_sample']:
                    b_points['b_point'].iloc[idx] = np.NaN
                    #warnings.warn(f"The detected B-Point is located before the R-Peak at heartbeat {idx}!"
                    #              f" The index of the B-Point was set to NaN. However, this should never happen!")
                else:
                    b_points['b_point'].iloc[idx] = b_point
            else:
                b_points['b_point'].iloc[idx] = b_point
            '''
            b_points['b_point'].iloc[idx] = b_point

        warnings.warn(f"Could not detect a local minimum in the RC-interval in {counter} heartbeats!")
        points = b_points
        self.points_ = points
        return self
