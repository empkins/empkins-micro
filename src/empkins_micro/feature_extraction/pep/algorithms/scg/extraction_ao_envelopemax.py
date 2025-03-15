from typing import Optional

import numpy as np
import pandas as pd
from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction
from empkins_micro.preprocessing.scg.envelope_construction import scg_norm, scg_lowpass, scg_highpass, scg_envelope
from tpcp import Parameter, make_action_safe


class AOExtraction_EnvelopeMax(BaseExtraction):

    # input parameters
    save_intermediate_results: Parameter[bool]

    # constants
    AO_SEARCH_WINDOW_ms = 200

    # results
    intermediate_results_: pd.DataFrame

    def __init__(
            self,
            save_intermediate_results: Optional[bool] = False
    ):
        """initialize new AOExtraction_EnvelopeMax algorithm instance (AO = aortic valve opening)

        Args:
            save_intermediate_results: bool
                indicates whether only the AO-point position (one per heartbeat) is saved in _points (False),
                or also the results of the processing steps are saved (True)
        """

        self.save_intermediate_results = save_intermediate_results

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts AO points from SCG signal

        aortic valve opening (AO) point corresponds to the maximum of the Hilbert envelope,
        first, the norm of the signal is calculated, then the norm is high-pass filtered (20 Hz cutoff),
        then, the Hilbert envelope is calculated,
        then, the envelope is squared and low-pass filtered (20 Hz cutoff),
        AO is defined as the max of the resulting envelope within the search window,
        method from Ahmaniemi 2019

        Args:
            signal_clean:
                pd.DataFrame with cleaned 3 channel acc signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of SCG signal in hz

        Returns:
            saves resulting AO-point positions in points_, index is heartbeat id
        """

        # create result df
        ao_points = pd.DataFrame(index=heartbeats.index, columns=["ao_point"])
        if self.save_intermediate_results:
            self.intermediate_results_ = pd.DataFrame(columns=["norm", "norm_hp", "envelope"], index=signal_clean.index)

        # calculate norm of all acc channels (scg_norm checks whether input signal has 3 channels)
        acc_norm = scg_norm(xyz_acc_signals=signal_clean)

        # high-pass filtering of signal norm
        acc_norm_hp = scg_highpass(scg_signal=acc_norm, sampling_rate_hz=sampling_rate_hz)

        # calculate Hilbert envelope of high-pass filtered signal, square envelope, low-pass filter envelope
        envelope = scg_envelope(scg_signal=acc_norm_hp)
        envelope = envelope**2
        envelope = scg_lowpass(scg_signal=envelope, sampling_rate_hz=sampling_rate_hz)

        if self.save_intermediate_results:
            self.intermediate_results_["norm"] = acc_norm
            self.intermediate_results_["norm_hp"] = acc_norm_hp
            self.intermediate_results_["envelope"] = envelope

        # calculate window length in samples
        window_length_samples = int((self.AO_SEARCH_WINDOW_ms / 1000) * sampling_rate_hz)

        # search AO for each heartbeat of the calculated envelope signal
        for idx, data in heartbeats.iterrows():

            # slice signal for current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]
            heartbeat_envelope = envelope.iloc[heartbeat_start:heartbeat_end]

            # calculate R-peak position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start

            # extract search window for this heartbeat
            window_start = heartbeat_r_peak
            window_end = window_start + window_length_samples
            window_envelope = heartbeat_envelope[window_start:window_end]

            # find max of envelope, which is AO
            envelope_max = np.argmax(window_envelope)

            # calculate position of AO relative to signal start and save AO in result
            ao_point = envelope_max + window_start + heartbeat_start
            ao_points.at[idx, "ao_point"] = ao_point

            self.points_ = ao_points

        return self
