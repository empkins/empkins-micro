from typing import Optional

import pandas as pd
import numpy as np
import neurokit2 as nk

from tpcp import Algorithm, Parameter, make_action_safe


class HeartBeatExtraction(Algorithm):
    """segment ECG signal into heartbeats"""

    _action_methods = "extract"

    # input parameters
    variable_length: Parameter[bool]
    start_factor: Parameter[float]

    # result
    heartbeat_list_: pd.DataFrame

    def __init__(
            self,
            variable_length: bool,
            start_factor: Optional[float] = 0.35
    ):
        """initialize new HeartBeatExtraction algorithm instance

        Parameters
        ----------
        variable_length : bool
            defines, if extracted heartbeats should have variable length (depending on the current RR-interval) or
            fixed length (same length for all heartbeats, depending on mean heartrate of the complete signal, 35% of
            mean heartrate in seconds before R-peak and 50% after r_peak, see neurokit2 ecg_segments)
            for variable length heartbeats, start of next heartbeat follows directly after end of last (ends exclusive)
            for fixed length heartbeats, there might be spaces between heartbeat boarders, or they might overlap
        start_factor : float, optional
            only needed for variable_length heartbeats, factor between 0 and 1, which defines where the start boarder
            between heartbeats is set depending on the RR-interval to previous heartbeat, for example factor 0.35 means
            that beat start is set at 35% of current RR-distance before the R-peak of the beat
        """

        self.variable_length = variable_length
        self.start_factor = start_factor

    @make_action_safe
    def extract(self, ecg_clean: pd.Series, sampling_rate_hz: int):
        """segments ecg signal into heartbeats, extract start, end, r-peak of each heartbeat

        fills df containing all heartbeats, one row corresponds to one heartbeat;
        for each heartbeat, df contains: start datetime, sample index of start/end, and sample index of r-peak;
        index of df can be used as heartbeat id

        Args:
            ecg_clean : containing cleaned ecg signal as pd series with datetime index
            sampling_rate_hz : containing sampling rate of ecg signal in hz as int
        Returns:
            self: fills heartbeat_list_
        """

        # TODO methode r peaks? neurokit, promac, ... ?
        # TODO correct artifacts?

        _, r_peaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate_hz, method="neurokit")
        r_peaks = r_peaks["ECG_R_Peaks"]

        idx = pd.RangeIndex.from_range(range(0, len(r_peaks)))
        heartbeats = pd.DataFrame(index=idx, columns=["heartbeat_start_time", "heartbeat_start_sample",
                                                      "heartbeat_end_sample", "r_peak_sample"])
        heartbeats["r_peak_sample"] = r_peaks

        if self.variable_length:
            # split ecg signal into heartbeats with varying length

            rr_interval_samples = heartbeats["r_peak_sample"].diff()

            # calculate start of each heartbeat based on corresponding R-peak and current RR-interval
            beat_starts = heartbeats["r_peak_sample"] - self.start_factor * rr_interval_samples

            # extrapolate first beats start based on RR-interval of next beat
            first_beat_start = heartbeats["r_peak_sample"].iloc[0] - self.start_factor * rr_interval_samples.iloc[1]
            if first_beat_start >= 0:
                beat_starts.iloc[0] = first_beat_start
            else:
                beat_starts = beat_starts.iloc[1:].reset_index(drop=True)  # drop row, when heartbeat is incomplete
                heartbeats = heartbeats.iloc[1:].reset_index(drop=True)
            beat_starts = round(beat_starts).astype(int)
            heartbeats["heartbeat_start_sample"] = beat_starts

            # calculate beat ends (last beat ends 1 sample before next starts, end is exclusive)
            beat_ends = beat_starts.shift(-1)  # end is exclusive

            # extrapolate last beats end based on RR-interval of previous beat
            last_beat_end = round(
                heartbeats["r_peak_sample"].iloc[-1] + (1 - self.start_factor) * rr_interval_samples.iloc[-1])
            if last_beat_end < len(ecg_clean):
                beat_ends.iloc[-1] = last_beat_end
            else:
                beat_ends = beat_ends.iloc[:-1]  # drop row, when heart beat is incomplete
                heartbeats = heartbeats.iloc[:-1]
            beat_ends = beat_ends.astype(int)
            heartbeats["heartbeat_end_sample"] = beat_ends

            # extract time of each beat's start
            beat_starts_time = ecg_clean.iloc[heartbeats["heartbeat_start_sample"]].index
            heartbeats["heartbeat_start_time"] = beat_starts_time

        else:
            # split ecg signal into heartbeats with fixed length

            heartbeat_segments = nk.ecg_segment(ecg_clean, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, show=False)
            for segment_idx in heartbeat_segments.keys():
                # extract sample number of start, end, r peak, and datetime of start from current segment
                segment = heartbeat_segments[segment_idx].reset_index(drop=True)
                start = segment["Index"].iloc[0]
                end = segment["Index"].iloc[-1]
                start_time = ecg_clean.index[start]

                # fill the corresponding row of heartbeats for current segment
                # (idx-1 because segments keys start with 1, but heartbeats_list should start with 0)
                heartbeats["heartbeat_start_sample"].iloc[int(segment_idx) - 1] = start
                heartbeats["heartbeat_end_sample"].iloc[int(segment_idx) - 1] = end
                heartbeats["heartbeat_start_time"].iloc[int(segment_idx) - 1] = start_time

        # check if R-peak occurs between corresponding start and end
        check = heartbeats.apply(lambda x: x["heartbeat_start_sample"] < x["r_peak_sample"] < x["heartbeat_end_sample"],
                                 axis=1)
        if len(check[check == False]) > 0:
            raise ValueError(
                f"Start, end, or r-peak position of heartbeat {list(check[check == False].index)} could be incorrect!")

        self.heartbeat_list_ = heartbeats
        return self
