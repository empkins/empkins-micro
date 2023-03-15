import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import argrelmin

from tpcp import Algorithm, Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction

import warnings


class BPointExtractionDrost(BaseExtraction):
    """algorithm to extract B-point based on the maximum distance of the dZ/dt curve and a straight line
    from the C-Point to the Point on the dZ/dt curve 150 ms before the C-Point"""

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

        # get the c_point locations and check if any entries contain NaN
        c_points = c_points['c_point']
        check_c_points = np.isnan(c_points.values.astype(float))

        # go trough each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_c_points[idx]:
                b_points['b_point'].iloc[idx] = np.NaN
                warnings.warn(f"Either the r_peak or the c_point contains NaN at position{idx}! "
                              f"B-Point was set to NaN.")
                continue
            else:
                # set the borders of the interval between the R-Peak and the C-Point
                c_point = c_points[idx]

            # Get the start_position of the straight line 150 ms before the C-Point
            line_start = c_point - int((150/1000) * sampling_rate_hz)

            # Compute the values of the straight line
            line_values = self.get_line_values(line_start, signal_clean[line_start], c_point, signal_clean[c_point])

            # Get the interval of the cleaned ICG-signal which matches the position of the straight line
            signal_clean_interval = signal_clean[line_start:c_point]

            # Compute the distance between the straight line and the cleaned ICG-signal
            distance = line_values['line_values'].values - signal_clean_interval.values

            # Select the position with maximal distance as the B-Point and convert it to absolute position
            b_point = distance.argmax() + line_start

            b_points['b_point'].iloc[idx] = b_point

        points = b_points
        self.points_ = points
        return self

    @staticmethod
    def get_line_values(start_x: int, start_y: float, c_x: int, c_y: float):
        """function which computes the values of a straight line between the C-Point and the Point 150 ms before
        the C-Point

        Args:
            start_x:
                int index of the Point 150 ms before the C-Point as an absolute index
            start_y:
                float value of the Point 150 ms before the C-Point
            c_x:
                int index of the C-Point
            c_y:
                float value of the C-Point

        Returns:
            pd.DataFrame with the values of the straight line for each index between the C-Point and the Point 150 ms
            before the C-Point
        """
        # Compute the gradient of the straight line
        m = (c_y - start_y) / (c_x - start_x)

        # Get the indexes where we want to compute the values of the straight line
        index = np.arange(0, (c_x - start_x), 1)
        line_values = pd.DataFrame(index, columns=['line_values'])

        # Compute the values of the straight line for each position in index
        for x in index:
            y = (m * x) + start_y
            line_values['line_values'].loc[x] = y
        return line_values
