import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import argrelmin
from itertools import tee

from tpcp import Algorithm, Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction
from empkins_micro.feature_extraction.pep.algorithms.icg.extraction_c_point_scipy_findpeaks import \
    CPointExtraction_ScipyFindPeaks

import warnings


class BPointExtractionForouzanfar(BaseExtraction):
    """algorithm to extract B-point based on [Forouzanfar et al., 2018, Psychophysiology]"""

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts B-points from given ICG cleaned signal

        Args:
            signal_clean:
                cleaned ICG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns:
            saves resulting B-point locations (samples) in points_ attribute of super class,
            index is C-point (/heartbeat) id
        """
        # Create the B-Point/A-Point Dataframes with the index of the heartbeat_list
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point"])
        intermediate_steps = pd.DataFrame(index=heartbeats.index, columns=["a_point", "height"])
        # We don't have to store the A-values

        # get the c_point locations and check if entries contain NaN
        c_points = self.get_c_points(signal_clean, heartbeats, sampling_rate_hz)
        check_c_points = np.isnan(c_points.values.astype(float))

        # search for the A-Point within one third of the beat to beat interval prior to the A-Point

        # go trough each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():   # use shift alternatively
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_c_points[idx]:
                b_points['b_point'].iloc[idx] = np.NaN
                warnings.warn(f"Either the r_peak or the c_point contains NaN at position{idx}! "
                              f"B-Point was set to NaN.")
                continue
            else:
                # get the actual and following R-Peak and the C-Point
                r_peak_start = data['r_peak_sample']
                r_peak_end = data.shift(1)['r_peak_sample']
                # Step 1: Detect the main peak in the dZ/dt signal (C-Point)
                c_point = c_points[idx]

            # Compute the beat to beat interval
            beat_to_beat = r_peak_end - r_peak_start

            # Compute the search interval for the A-Point
            search_interval = int(beat_to_beat/3)

            # Step 2: Detect the local minimum (A-Point) within one third of the beat to beat interval
            a_point = self.get_a_point(signal_clean, search_interval, c_point) + (c_point - search_interval)
            intermediate_steps['a_point'].iloc[idx] = a_point

            # Step 3: Calculate the amplitude difference between the C-Point and the A-Point
            height = signal_clean[c_point] - signal_clean[a_point]
            intermediate_steps['height'].iloc[idx] = height

            # Get the signal_segment between the A-Point and the C-Point
            signal_clean_segment = signal_clean[a_point:c_point+1]

            # Step 4.1: Get all monotonic increasing segments between the A-Point and the C-Point
            start_indexes, end_indexes = self.get_monotonic_increasing_segments(signal_clean_segment) + (c_point - search_interval)

            #b_points['b_point'].iloc[idx] = b_point

        points = b_points
        self.points_ = points
        return self

    @staticmethod
    def get_c_points(signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):

        c_point_extractor = CPointExtraction_ScipyFindPeaks(window_c_correction=3, save_candidates=False)
        c_point_extractor.extract(signal_clean=signal_clean, heartbeats=heartbeats, sampling_rate_hz=sampling_rate_hz)
        c_points = c_point_extractor.points_

        return c_points

    @staticmethod
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    @staticmethod
    def get_a_point(signal_clean: pd.DataFrame, search_interval: int, c_point: int):
        signal_interval = signal_clean[(c_point - search_interval):c_point]
        signal_minima = argrelmin(signal_interval.values, mode='wrap')
        print(signal_minima)
        a_point_idx = np.argmin(signal_interval.iloc[signal_minima[0]])
        a_point = signal_minima[0][a_point_idx]
        return a_point

    @staticmethod
    def get_monotonic_increasing_segments(signal_clean_segment: pd.DataFrame):
        signal_clean_segment.index = np.arange(0, len(signal_clean_segment))
        change_in_monotony = signal_clean_segment.diff().fillna(0)
        change_in_monotony['borders'] = 0

        # if there is a change from True to False, which means that the gradient changes from negative to positive,
        # insert 'start_increase' in the borders column
        change_in_monotony.loc[((change_in_monotony['icg'][1:] >= 0) &
                               (change_in_monotony['icg'].shift(1) < 0)), 'borders'] = 'start_increase'
        # vice versa
        change_in_monotony.loc[((change_in_monotony['icg'][1:] >= 0) &
                               (change_in_monotony['icg'].shift(-1) < 0)), 'borders'] = 'end_increase'

        # Since the end_point of the signal_segment is the C-Point, we have to insert end_increase in the borders column
        if change_in_monotony['icg'][len(change_in_monotony) - 1] >= 0:
            change_in_monotony['borders'].iloc[-1] = 'end_increase'

        # Since the first point of the segment is the A-Point (global minimum in this segment), we have to insert
        # start_increase in the borders column
        if change_in_monotony['icg'][1] >= 0:
            change_in_monotony['borders'].iloc[0] = 'start_increase'

        start_indexes = change_in_monotony.index[change_in_monotony['borders'] == 'start_increase'].to_list()
        end_indexes = change_in_monotony.index[change_in_monotony['borders'] == 'end_increase'].to_list()
        if signal_clean_segment.iloc[start_indexes] < (0.5 * signal_clean_segment.iloc[-1]):

        return start_indexes, end_indexes       # That are not absolute positions yet

