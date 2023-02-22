import pandas as pd
import numpy as np
import neurokit2 as nk

from tpcp import Algorithm, Parameter, make_action_safe


class HeartBeatExtraction(Algorithm):

    _action_methods = "extract"

    # result
    heartbeat_list_: pd.DataFrame

    # input cleaned ecg signal as pd series
    @make_action_safe
    def extract(self, ecg_clean: pd.Series, sampling_rate_hz: int):
        """segments ecg signal into heartbeats, extract start, end, r-peak of each heartbeat

        fills df containing all heartbeats, one row corresponds to one heartbeat;
        for each heartbeat, df contains: start datetime, sample index of start/end, and sample index of r-peak;
        index of df can be used as heartbeat id

        Args:
            ecg_clean: cleaned ecg signal as pd series with datetime index
            sampling_rate_hz: sampling rate of ecg signal in hz as int
        Returns:
            self: fills heartbeat_list_
        """

        # TODO methode r peaks? neurokit, promac, ... ?

        r_peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate_hz, method="neurokit")

        # split ecg signal into heartbeats (the next heartbeat does not necessarily start where the previous one ends?!)
        segments = nk.ecg_segment(ecg_clean, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate_hz, show=True)

        # create result df, length is number of heartbeat segments
        heartbeat_df = pd.DataFrame(index=np.arange(len(segments)), columns=["heartbeat_start_time",
                                                                             "heartbeat_start_sample",
                                                                             "heartbeat_end_sample",
                                                                             "r_peak_sample"])

        for segment_idx in segments.keys():

            # extract sample number of start, end, r peak, and datetime of start from current segment
            segment = segments[segment_idx].reset_index(drop=True)
            start = segment.iloc[0]["Index"]
            end = segment.iloc[-1]["Index"]
            start_time = ecg_clean.index[start]

            # extract r-peak sample number from nk ecg_peaks result (indexing starts with 0)
            r_peak = info["ECG_R_Peaks"][int(segment_idx) - 1]

            # r-peak needs to occur in between start and end sample of current heartbeat, this makes sure that r-peaks
            # are associated with correct heartbeats (nothing shifted due to missing r-peaks, etc.)
            if not (start < r_peak < end):
                raise ValueError(f"Start, end, or r-peak position of heartbeat {segment_idx} could be incorrect!")

            # fill the corresponding row of heartbeat_list for current segment
            # (idx-1 because segments keys start with 1, but heartbeats_list should start with 0)
            heartbeat_df.iloc[int(segment_idx) - 1]["heartbeat_start_time"] = start_time
            heartbeat_df.iloc[int(segment_idx) - 1]["heartbeat_start_sample"] = start
            heartbeat_df.iloc[int(segment_idx) - 1]["heartbeat_end_sample"] = end
            heartbeat_df.iloc[int(segment_idx) - 1]["r_peak_sample"] = r_peak

        self.heartbeat_list_ = heartbeat_df
        return self

