import warnings
from typing import Optional

import numpy as np
import pandas as pd
from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction
from scipy import signal
from tpcp import Parameter, make_action_safe


class AOExtraction_AccZ(BaseExtraction):

    # input parameters
    save_ivc: Parameter[bool]

    # constants
    AO_SEARCH_WINDOW_ms = 200

    def __init__(
            self,
            save_ivc: Optional[bool] = False
    ):
        """initialize new AOExtraction algorithm instance (AO = aortic valve opening)

        Args:
            save_ivc : bool
                indicates whether only the AO-point position (one per heartbeat) is saved in _points (False),
                or also the IVC based on which the AO was selected (True)
        """

        self.save_ivc = save_ivc

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts AO points from SCG signal

        aortic valve opening point corresponds to the first major maximum of the SCG signal that follows after the
        isovolumetric contraction (IVC), which is the first major minimum of the AO signal,
        method is modified from Di Rienzo 2017 & Ahmaniemi 2019

        Args:
            signal_clean:
                cleaned SCG signal or norm of cleaned x, y, z channels
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of SCG signal in hz

        Returns:
            saves resulting AO-point positions (and corresponding IVC) in points_, index is heartbeat id
        """

        # result df
        ao_points = pd.DataFrame(index=heartbeats.index, columns=["ao_point", "ivc"])
        if not self.save_ivc:
            ao_points.drop(columns="ivc", inplace=True)

        # calculate window length in samples
        window_length_samples = int((self.AO_SEARCH_WINDOW_ms / 1000) * sampling_rate_hz)

        # used subsequently to store ids of heartbeats for which no AO or IVC could be detected
        heartbeats_no_ao = []
        heartbeats_no_ivc = []

        # search AO for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():

            # slice signal for current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]
            heartbeat_scg = signal_clean.iloc[heartbeat_start:heartbeat_end]

            # calculate R-peak position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start

            # extract search window for this heartbeat
            window_start = heartbeat_r_peak
            window_end = window_start + window_length_samples
            window_scg = heartbeat_scg[window_start:window_end]

            # TODO? evtl kann man auch so ähnlich wie bei C's correcten, aber glaub nicht, dass das nötig ist
            # find possible IVCs (minima in search window)
            ivc_candidates = signal.find_peaks(-window_scg)[0]

            # discard minima which are not "prominent" enough to be IVC
            # IVC points should have a height > 0.5 * amplitude of the lowest min with respect to the mean of window
            # (other minima might be smaller probably random reflections)
            window_mean = np.mean(window_scg)
            window_min = np.min(window_scg)
            threshold_ivc = 0.5 * np.abs(window_min - window_mean)  # positive distance
            ivc_candidates = [ivc for ivc in ivc_candidates if window_scg.iloc[ivc] < (window_mean - threshold_ivc)]  # ivc lower than mean-threshold

            # no IVC found, consequently no valid AO can be found, continue with next heartbeat
            if len(ivc_candidates) < 1:
                heartbeats_no_ivc.append(idx)
                heartbeats_no_ao.append(idx)
                ao_points.at[idx, "ao_point"] = np.NaN
                if self.save_ivc:
                    ao_points.at[idx, "ivc"] = np.NaN
                continue

            # possible IVCs were found, select first one
            ivc = ivc_candidates[0]

            # find possible AOs (maxima in search window)
            ao_candidates = signal.find_peaks(window_scg)[0]

            # no AO candidates found, or none of the AO candidates occurs after IVC, continue with next heartbeat
            if len(ao_candidates) < 1 or ao_candidates[-1] <= ivc:
                heartbeats_no_ao.append(idx)
                ao_points.at[idx, "ao_point"] = np.NaN
                if self.save_ivc:
                    ivc += window_start + heartbeat_start
                    ao_points.at[idx, "ivc"] = ivc
                continue

            # possible AOs were found, only keep the ones occurring after IVC
            ao_candidates = [ao for ao in ao_candidates if ao > ivc]

            # AOs should be significantly higher than IVC, at least 0.7 * amplitude of IVC w.r.t window mean
            # discard all AOs which do not satisfy that criterion
            # (can happen due to smaller reflections in between IVC and real AO)
            ivc_value = window_scg.iloc[ivc]
            threshold_ao = 0.7 * np.abs(ivc_value - window_mean)  # positive distance
            ao_candidates = [ao for ao in ao_candidates if window_scg.iloc[ao] > (ivc_value + threshold_ao)]

            # check again if valid AO's were found
            if len(ao_candidates) < 1:
                heartbeats_no_ao.append(idx)
                ao_points.at[idx, "ao_point"] = np.NaN
                if self.save_ivc:
                    ivc += window_start + heartbeat_start
                    ao_points.at[idx, "ivc"] = ivc
                continue

            # select first one (closest after IVC)
            ao = ao_candidates[0]

            # calculate position of AO-point relative to signal start
            ao_point = ao + window_start + heartbeat_start
            ao_points.at[idx, "ao_point"] = ao_point
            if self.save_ivc:
                ivc += window_start + heartbeat_start
                ao_points.at[idx, "ivc"] = ivc

        if len(heartbeats_no_ivc) > 0:
            warnings.warn(f"No IVC-point detected in {len(heartbeats_no_ivc)} heartbeats ({heartbeats_no_ivc})")
        if len(heartbeats_no_ao) > 0:
            warnings.warn(f"No AO-point detected in {len(heartbeats_no_ao)} heartbeats ({heartbeats_no_ao})")

        self.points_ = ao_points
        return self
