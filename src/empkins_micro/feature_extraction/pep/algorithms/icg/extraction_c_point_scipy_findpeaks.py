import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tpcp import Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction


class CPointExtraction_ScipyFindPeaks(BaseExtraction):
    """algorithm to extract C-points from ICG derivative signal using scipy's find_peaks function"""

    # input parameters
    window_c_correction: Parameter[int]
    save_candidates: Parameter[bool]

    def __init__(self, window_c_correction: Optional[int] = 3, save_candidates: Optional[bool] = False):
        """initialize new CPointExtraction_ScipyFindPeaks algorithm instance
        Args:
            window_c_correction : int
                how many preceding heartbeats are taken into account for C-point correction (using mean R-C-distance)
            save_candidates : bool
                indicates whether only the selected C-point position (one per heartbeat) is saved in _points (False),
                or also all other C-candidates (True)
        """

        self.window_c_correction = window_c_correction
        self.save_candidates = save_candidates

    @make_action_safe
    def extract(self, signal_clean: pd.Series, heartbeats: pd.DataFrame, sampling_rate_hz: int):
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
            saves resulting C-point positions (and C-candidates) in points_, index is heartbeat id
        """

        # result df
        c_points = pd.DataFrame(index=heartbeats.index, columns=["c_point", "c_candidates"])
        if self.save_candidates:
            c_points["c_candidates"] = c_points.apply(lambda x: [], axis=1)
        else:
            c_points.drop(columns="c_candidates", inplace=True)

        # distance of R-peak to C-point, averaged over as many preceding heartbeats as window_c_correction specifies
        # R-C-distances are positive when C-point occurs after R-Peak (which is the physiologically correct order)
        mean_prev_r_c_distance = np.NaN

        # saves R-C-distances of previous heartbeats
        prev_r_c_distances = []

        # used subsequently to store heartbeats for which no C-point could be detected
        heartbeats_no_c = []

        # search C-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():
            # slice signal for current heartbeat

            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]

            heartbeat_icg_der = signal_clean.iloc[heartbeat_start:heartbeat_end]

            # calculate R-peak position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start
            prominence = 1.0
            # detect possible C-point candidates, prominence=1 gives reasonable Cs
            # (prominence=2 results in a considerable amount of heartbeats with no C)
            # (prominence=1 might detect more than one C in one heartbeat, but that will be corrected subsequently)
            heartbeat_c_candidates = signal.find_peaks(heartbeat_icg_der, prominence=prominence)[0]

            while len(heartbeat_c_candidates) < 1 and prominence > 0.1:
                prominence -= 0.1
                heartbeat_c_candidates = signal.find_peaks(heartbeat_icg_der, prominence=prominence)[0]

            if len(heartbeat_c_candidates) < 1:
                heartbeats_no_c.append(idx)
                c_points.at[idx, "c_point"] = np.NaN
                continue

            # calculates distance of R-peak to all C-candidates in samples, positive when C occurs after R
            r_c_distance = heartbeat_c_candidates - heartbeat_r_peak

            if len(heartbeat_c_candidates) == 1:
                selected_c = heartbeat_c_candidates[0]  # convert to int (instead of array)
                r_c_distance = r_c_distance[0]

                # C-point before R-peak is invalid
                if r_c_distance < 0:
                    heartbeats_no_c.append(idx)
                    c_points.at[idx, "c_point"] = np.NaN
                    continue

            elif len(heartbeat_c_candidates) > 1:
                negative_indices = np.where(r_c_distance < 0)[0]

                r_c_distance = r_c_distance[r_c_distance >= 0]
                heartbeat_c_candidates = np.delete(heartbeat_c_candidates, negative_indices)
                # take averaged R-C-distance over the 3 (or window_c_correction) preceding heartbeats
                # calculate the absolute difference of R-C-distances for all C-candidates to this mean
                # (to check which of the C-candidates are most probably the wrongly detected Cs)

                if len(r_c_distance) > 1:
                    distance_diff = np.abs(r_c_distance - mean_prev_r_c_distance)

                    # choose the C-candidate with the smallest absolute difference in R-C-distance
                    # (the one, where R-C-distance changed the least compared to previous heartbeats)
                    c_idx = np.argmin(distance_diff)
                    selected_c = heartbeat_c_candidates[c_idx]
                    r_c_distance = r_c_distance[c_idx]  # save only R-C-distance for selected C
                elif len(r_c_distance) == 0:
                    heartbeats_no_c.append(idx)
                    c_points.at[idx, "c_point"] = np.NaN
                    continue
                else:
                    selected_c = heartbeat_c_candidates[0]  # convert to int (instead of array)
                    r_c_distance = r_c_distance[0]

            else:
                warnings.warn("That should never happen!")
                selected_c = np.NaN

            # update R-C-distances and mean for next heartbeat
            prev_r_c_distances.append(r_c_distance)
            if len(prev_r_c_distances) > self.window_c_correction:
                prev_r_c_distances.pop(0)
            mean_prev_r_c_distance = np.mean(prev_r_c_distances)

            # save C-point (and C-candidates) to result property
            c_points.at[idx, "c_point"] = selected_c + heartbeat_start  # get C-point relative to complete signal
            if self.save_candidates:
                for c in heartbeat_c_candidates:
                    c_points.at[idx, "c_candidates"].append(c + heartbeat_start)

        if len(heartbeats_no_c) > 0:
            warnings.warn(f"No valid C-point detected in {len(heartbeats_no_c)} heartbeats ({heartbeats_no_c})")
       
        
        self.points_ = c_points
        return self
