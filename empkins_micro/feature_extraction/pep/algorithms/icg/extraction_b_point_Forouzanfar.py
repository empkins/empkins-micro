import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import argrelmin, argrelextrema
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

        # get the c_point locations and check if entries contain NaN
        c_points = self.get_c_points(signal_clean, heartbeats, sampling_rate_hz)
        check_c_points = np.isnan(c_points.values.astype(float))

        # Calculate the second and third derivative of the signal
        second_der = np.gradient(signal_clean)
        third_der = np.gradient(second_der)

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
                # Detect the main peak in the dZ/dt signal (C-Point)
                c_point = c_points['c_point'][idx]

            # Compute the beat to beat interval
            beat_to_beat = r_peak_end - r_peak_start

            # Compute the search interval for the A-Point
            search_interval = int(beat_to_beat/3)

            # Step 2: Detect the local minimum (A-Point) within one third of the beat to beat interval
            a_point = self.get_a_point(signal_clean, search_interval, c_point) + (c_point - search_interval)

            # Step 3: Calculate the amplitude difference between the C-Point and the A-Point
            height = signal_clean.iloc[c_point] - signal_clean.iloc[a_point]

            # Get the signal_segment between the A-Point and the C-Point
            signal_clean_segment = signal_clean.iloc[a_point:c_point+1]

            # Step 4.1: Get the most prominent monotonic increasing segment between the A-Point and the C-Point
            start_sample, end_sample = self.get_monotonic_increasing_segments(signal_clean_segment, height) + (c_point - search_interval)
            print(f"Monotonic increasing segments start_sample {start_sample}, end_sample {end_sample}, heartbeat_id {idx}.")
            # Get the first third of the monotonic increasing segment
            start = start_sample
            end = end_sample - int(2 * (end_sample - start_sample) / 3)
            print(f"First third of the monotonic segment, start: {start} end: {end}")

            # 2nd derivative of the segment
            monotonic_segment_2nd_der = pd.DataFrame(second_der[start:end], columns=['2nd_der'])
            # 3rd derivative of the segment
            monotonic_segment_3rd_der = pd.DataFrame(third_der[start:end], columns=['3rd_der'])

            # Compute the significant zero_crossings
            significant_zero_crossings = self.get_zero_crossings(
                monotonic_segment_3rd_der, monotonic_segment_2nd_der, height, sampling_rate_hz)

            # Compute the significant local maximums of the 3rd derivative of the most prominent monotonic segment
            significant_local_maximums = self.get_local_maximums(monotonic_segment_3rd_der, height, sampling_rate_hz)

            # Label the last zero crossing/ local maximum as the B-Point
            # If there are no zero crossings or local maximums use the first Point of the segment as B-Point
            if isinstance(significant_zero_crossings, type(None)) & isinstance(significant_local_maximums, type(None)):
                b_points['b_point'].iloc[idx] = start_sample
            else:
                significant_features = pd.concat([significant_zero_crossings, significant_local_maximums], axis=0)
                if len(significant_features.index) == 0:
                    b_points['b_point'].iloc[idx] = start_sample
                else:
                    b_point = significant_features.iloc[np.argmin(c_point - significant_features)]
                    b_points['b_point'].iloc[idx] = b_point

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
    def get_a_point(signal_clean: pd.DataFrame, search_interval: int, c_point: int):
        signal_interval = signal_clean.iloc[(c_point - search_interval):c_point]
        signal_minima = argrelmin(signal_interval.values, mode='wrap')
        a_point_idx = np.argmin(signal_interval.iloc[signal_minima[0]])
        a_point = signal_minima[0][a_point_idx]
        return a_point

    @staticmethod
    def get_monotonic_increasing_segments(signal_clean_segment: pd.DataFrame, height: int):
        signal_clean_segment.index = np.arange(0, len(signal_clean_segment))
        monotony_df = pd.DataFrame(signal_clean_segment, columns=['icg'])
        monotony_df['diff'] = monotony_df.diff().fillna(0)
        monotony_df['borders'] = 0

        # start_increase if there is a change from negative to positive
        monotony_df.loc[((monotony_df['diff'][1:] >= 0) &
                               (monotony_df['diff'].shift(1) < 0)), 'borders'] = 'start_increase'
        # end_increase if there is a change from positive to negative
        monotony_df.loc[((monotony_df['diff'][1:] >= 0) &
                               (monotony_df['diff'].shift(-1) < 0)), 'borders'] = 'end_increase'

        # Since the end_point of the signal_segment is the C-Point, we have to insert end_increase in the borders column
        if monotony_df['diff'][len(monotony_df) - 1] >= 0:
            monotony_df['borders'].iat[-1] = 'end_increase'

        # Since the first point of the segment is the A-Point (global minimum in this segment), we have to insert
        # start_increase in the borders column
        if monotony_df['diff'][1] >= 0:
            monotony_df['borders'].iat[0] = 'start_increase'
        # drop all samples that are no possible start-/ end-points
        monotony_df = monotony_df.drop(monotony_df[monotony_df['borders'] == 0].index)
        monotony_df = monotony_df.reset_index()
        if len(monotony_df) > 2:
            # Drop start- and corresponding end-point, if their start value does not reach at least 1/2 of H
            monotony_df = monotony_df.drop(
                monotony_df[(monotony_df['borders'] == 'start_increase') & (monotony_df['icg'] > int(height/2))].index + 1)           # How is the C-Point amplitude defined?
            monotony_df = monotony_df.drop(
                monotony_df[(monotony_df['borders'] == 'start_increase') & (monotony_df['icg'] > int(height/2))].index)
            # Drop start- and corresponding end-point, if their end values does not reach at least 2/3 of H
            monotony_df = monotony_df.drop(
                monotony_df[(monotony_df['borders'] == 'end_increase') & (monotony_df['icg'] < int(2*height/3))].index - 1)
            monotony_df = monotony_df.drop(
                monotony_df[(monotony_df['borders'] == 'end_increase') & (monotony_df['icg'] < int(2*height/3))].index)

        # If their is more than one combination of points remaining, select the pair of points, that has the highest
        # amplitude difference
        start_sample = 0
        end_sample = 0
        monotony_df['icg'].diff().fillna(0)
        if len(monotony_df) > 2:
            idx = np.argmax(monotony_df['icg'].diff().fillna(0) > 0)
            start_sample = monotony_df['index'].iloc[idx-1]
            end_sample = monotony_df['index'].iloc[idx]
        elif len(monotony_df) == 0:
            warnings.warn("Could not find a monotonic segment. This should never happen!")
        else:
            start_sample = monotony_df['index'].iloc[0]
            end_sample = monotony_df['index'].iloc[-1]

        return start_sample, end_sample     # That are not absolute positions yet

    @staticmethod
    def get_zero_crossings(monotonic_segment_3rd_der: pd.DataFrame, monotonic_segment_2nd_der: pd.DataFrame, height: int, sampling_rate_hz: int):
        zero_crossings = np.where(np.diff(np.signbit(monotonic_segment_3rd_der['3rd_der'])))[0]
        zero_crossings = pd.DataFrame(zero_crossings, columns=['zero_crossings'])
        # Discard zero_crossings with negative to positive sig change
        significant_crossings = zero_crossings.drop(
            zero_crossings[monotonic_segment_2nd_der.iloc[zero_crossings['zero_crossings']] < 0].index, axis=0)
        # Discard zero crossings with slope higher than 10*H/f_s
        significant_crossings = significant_crossings.drop(
            significant_crossings[monotonic_segment_2nd_der.iloc[significant_crossings['zero_crossings']] >
                           10 * height / sampling_rate_hz].index, inplace=True, axis=0)
        return significant_crossings

    @staticmethod
    def get_local_maximums(monotonic_segment_3rd_der: pd.DataFrame, height: int, sampling_rate_hz: int):
        local_maximums = argrelextrema(monotonic_segment_3rd_der['3rd_der'].values, np.greater_equal)[0]
        local_maximums = pd.DataFrame(local_maximums, columns=['local_maximums'])
        significant_maximums = local_maximums.drop(local_maximums[
            monotonic_segment_3rd_der.iloc[
                local_maximums['local_maximums']].values < 4 * height / sampling_rate_hz].index, axis=0)
        return significant_maximums

