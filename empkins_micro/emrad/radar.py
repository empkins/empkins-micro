# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 16:04:08 2022

@author: nonev
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
import biopsykit as bp
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view
import neurokit2 as nk

from empkins_micro.emrad.utils import input_conversion, correct_peaks

Radar_Freq_Hz = 61 * 1e9  # 122 GHz
C_Light_Air_mps = 299708516  # reduzierte Lichtgeschwindigkeit in bodennaher Luft
Radian2Meter = C_Light_Air_mps / (4 * np.pi * Radar_Freq_Hz)
TIMESTEPS = 400
MAXBPM = 130
Minimumperiod = 60 / MAXBPM
DETECTIONTHRESHOLD = 0.05


def get_rpeaks(
    radar_data: pd.DataFrame, fs_radar: float, window_size: int
) -> bp.utils.datatype_helper.RPeakDataFrame:

    data_out = {}
    duration = (radar_data.index[-1] - radar_data.index[0]).total_seconds()
    num_windows = int(duration // window_size)
    print("------ duration ------")
    print(duration)
    print("------ num_windows before and after factoring overlap------")
    print(num_windows)
    #num_win for slinding window with overlap 50%
    overlap = 2
    num_windows = num_windows * overlap + 1
    if num_windows * window_size / overlap > duration:
        num_windows = num_windows - 1
    print(num_windows)

    processing = Processing(FS=fs_radar, window_size=window_size)

    for wind_ctr in range(num_windows):
        #shift the window by overlap of the window size
        start_sample_radar = round((wind_ctr * (window_size / overlap) * fs_radar))
        end_sample_radar = min(
            start_sample_radar + round((fs_radar * window_size)), len(radar_data)
        )


        radar_slice = radar_data.iloc[start_sample_radar:end_sample_radar]

        processing.setRadarSamples(radar_slice)
        processing.predictBeats()

        predicted_beats = processing.getBeats()[["predicted_beats"]]
        data_out[wind_ctr] = predicted_beats[int((window_size / (overlap * 2) * fs_radar)):int(-((window_size / (overlap * 2)) * fs_radar))]


    if len(data_out) > 1:
        data_concat = pd.concat(data_out, names=["participant", "window_id"])
    else:
        data_concat = pd.DataFrame(predicted_beats)

    radar_beats = find_peaks(
        data_concat.predicted_beats, height=0.08, distance=0.3 * fs_radar
    )[0]
    radar_beats = pd.DataFrame(
        radar_beats, index=data_concat.index[radar_beats], columns=["peak_idx"]
    )


    radar_beats["R_Peak_Quality"] = np.ones(
        len(radar_beats)
    )  # this does not make sense, but is required by biopsykit
    radar_beats["R_Peak_Outlier"] = np.zeros(
        len(radar_beats)
    )  # this does not make sense, but is required by biopsykit
    radar_beats.rename({"peak_idx": "R_Peak_Idx"}, axis=1, inplace=True)

    radar_beats["RR_Interval"] = (
        np.ediff1d(radar_beats["R_Peak_Idx"], to_end=0) / fs_radar
    )
    # ensure equal length by filling the last value with the average RR interval
    radar_beats.loc[radar_beats.index[-1], "RR_Interval"] = radar_beats[
        "RR_Interval"
    ].mean()

    bp.signals.ecg.EcgProcessor.correct_outlier(
        rpeaks=radar_beats,
        sampling_rate=fs_radar,
        imputation_type="moving_average",
        outlier_correction=["physiological", "statistical_rr", "statistical_rr_diff"],
    )

    return radar_beats, data_concat


def transform_for_nk_hrv(
        peaks: pd.DataFrame, lstm_output: pd.DataFrame, fs_radar: float
):
    pos_in_time = np.array([], dtype='object')
    peak_ind = np.array([], dtype='int64')

    for i in peaks['R_Peak_Idx']:
        if np.isnan(i):
            continue
        pos_in_time = np.append(pos_in_time, lstm_output.index[int(i)])
        peak_ind = np.append(peak_ind, int(i))

    time_to_int = np.array([], dtype='int64')
    val_peak = np.array([], dtype='int64')

    for i in range(len(lstm_output)):
        time_to_int = np.append(time_to_int, i)
        val_peak = np.append(val_peak, 0)
        j = 0
        if lstm_output.index[i] == pos_in_time[j]:
            val_peak[i] = 1
            j = j + 1

    info = {'method_peaks': 'emkins_micro',
            'method_fixpeaks': 'None',
            'ECG_R_Peaks': peak_ind,
            'sampling_rate': fs_radar}

    hrv_input_peak = pd.DataFrame({"ECG_R_Peaks": val_peak},
                                    index=time_to_int)

    return hrv_input_peak, info

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(12)
class TemporalFilters:
    # Funktionen Filter
    def butter_bandpass_coeff(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = scipy.signal.butter(
            order, [low, high], btype="band", analog=False, output="sos"
        )
        return sos

    def butter_lowpass_coeff(self, cutOff, fs, order=5):
        sos = scipy.signal.butter(
            order, cutOff, btype="lowpass", analog=False, output="sos", fs=fs
        )
        return sos

    def butter_highpass_coeff(self, cutOff, fs, order=5):
        sos = scipy.signal.butter(
            order, cutOff, btype="highpass", analog=False, output="sos", fs=fs
        )
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5, zi=None):
        sos = self.butter_bandpass_coeff(lowcut, highcut, fs, order=order)
        # y = filtfilt(b, a, data)
        y = scipy.signal.sosfiltfilt(sos, data)
        return y

    def butter_lowpass_filter(self, data, cutOff, fs, order=4, zi=None):
        sos = self.butter_lowpass_coeff(cutOff, fs, order=order)
        # y = filtfilt(b, a, data)
        y = scipy.signal.sosfiltfilt(sos, data)
        return y

    def butter_highpass_filter(self, data, cutOff, fs, order=4, zi=None):
        sos = self.butter_highpass_coeff(cutOff, fs, order=order)
        # y = filtfilt(b, a, data)
        y = scipy.signal.sosfiltfilt(sos, data)
        return y


class Processing:
    def create_dataset(self, X, time_steps=1, stepsize=1):

        Xs = []

        for i in range(0, len(X) - time_steps, stepsize):
            v = X[i : (i + time_steps)]

            Xs.append(v)

        return np.array(Xs)

    def __init__(self, FS, window_size):

        self.filter_fun = TemporalFilters()
        self.FS_Radar = FS
        self.window_size = window_size
        self.num_samples = round(self.FS_Radar * self.window_size)

        self.radarWindowedSamples = np.zeros((self.num_samples, 2))
        self.wrapped_raw_phase = np.zeros(self.num_samples)
        self.unwrapped_raw_phase = np.zeros(self.num_samples)
        self.FS_Radar = FS
        self.hs = np.zeros(self.num_samples)
        self.pw = np.zeros(self.num_samples)
        self.rad_i = np.zeros(self.num_samples)
        self.rad_q = np.zeros(self.num_samples)
        self.br = np.zeros(self.num_samples)

        # Decimated Signals for NN
        length_decimated = round(self.num_samples * 4)
        self.hs_envelope = np.zeros(length_decimated)
        self.rad_lp = np.zeros(length_decimated)
        self.rad_i_dc = np.zeros(length_decimated)
        self.rad_q_dc = np.zeros(length_decimated)
        self.angle = np.zeros(length_decimated)

        path = Path(__file__).parent / "my_model.keras"
        self.model = tf.keras.models.load_model(str(path))


        self.predictedBeats = np.zeros(self.num_samples)
        self.respBeats = []

    def predictBeats(self):

        self.angle = np.diff(np.unwrap(np.arctan2(self.rad_i, self.rad_q)), axis=0)

        # Radar Filtering and Feature Generation

        self.hs_envelope = np.convolve(
            np.abs(scipy.signal.hilbert(self.hs)).flatten(),
            np.ones(100) / 100,
            mode="same",
        )
        self.rad_lp = self.filter_fun.butter_lowpass_filter(self.rad, 15, self.FS_Radar)
        self.hs_envelope = scipy.signal.decimate(self.hs_envelope, 20, axis=0)

        self.rad_lp = scipy.signal.decimate(self.rad_lp, 20, axis=0)

        self.rad_i_dc = scipy.signal.decimate(self.rad_i, 20, axis=0)
        self.rad_q_dc = scipy.signal.decimate(self.rad_q, 20, axis=0)
        self.angle = scipy.signal.decimate(self.angle, 20, axis=0)

        # scales = np.arange(1,31);
        # coefs, _ = pywt.cwt(self.hs, np.arange(1,256),"gaus1");
        # print(np.shape(coefs))

        rad_vectors = np.transpose(
            np.vstack(
                (
                    self.rad_lp,
                    self.hs_envelope,
                    self.angle,
                    self.rad_i_dc,
                    self.rad_q_dc,
                )
            )
        )

        mean = rad_vectors.mean(axis=0)
        std = rad_vectors.std(axis=0)
        rad_vectors = (rad_vectors - mean) / std

        rad_vectors = np.nan_to_num(rad_vectors)

        Xs = []

        for i in range(0, len(rad_vectors) - TIMESTEPS, 20):
            v = rad_vectors[i : (i + TIMESTEPS)]

            Xs.append(v)
        X = np.array(Xs)

        predictions = np.squeeze(self.model.predict(X[:, :, (0, 1)]))
        reconstruct = np.zeros(len(predictions) * 20 + 20)

        n = 0
        for prediction in predictions:
            reconstruct[n : n + 20] = prediction[190:-190]
            n = n + 20

        reconstruction = np.pad(reconstruct, (190, 190))

        predicted_beats = scipy.signal.resample(
            reconstruction, len(self.radarWindowedSamples)
        )

        self.predictedBeats = pd.DataFrame(
            predicted_beats,
            columns=["predicted_beats"],
            index=self.radarWindowedSamples.index,
        )
        self.beatpeaks, _ = scipy.signal.find_peaks(
            predicted_beats,
            distance=Minimumperiod * self.FS_Radar,
            height=DETECTIONTHRESHOLD,
        )

        if not self.beatpeaks.any():  # No peaks detected
            self.beatpeaks = np.array([0])
        else:
            self.beatpeaks = np.array(np.around(self.beatpeaks / 2))
            temppeaks = input_conversion(
                self.beatpeaks.astype(int), input_type="peaks_idx", output_type="peaks"
            )
            corrpeaks = correct_peaks(
                temppeaks,
                input_type="peaks",
                missed_correction=False,
                extra_correction=False,
            )
            temp = np.where(corrpeaks["clean_peaks"])[0] * 2
            # temp = systole.utils.input_conversion(corrpeaks,input_type='peaks',output_type='peaks_idx')*2
            self.beatpeaks = temp

    def setRadarSamples(self, samples):

        self.radarWindowedSamples = samples

        # print(self.radarWindowedSamples[:, "I"])

        self.rad_i = self.filter_fun.butter_highpass_filter(
            self.radarWindowedSamples["I"], 0.2, self.FS_Radar
        )
        self.rad_q = self.filter_fun.butter_highpass_filter(
            self.radarWindowedSamples["Q"], 0.2, self.FS_Radar
        )

        self.rad = np.sqrt(np.square(self.rad_i) + np.square(self.rad_q))
        self.hs = self.filter_fun.butter_bandpass_filter(
            self.rad, 15, 60, self.FS_Radar
        )
        self.pw = self.filter_fun.butter_lowpass_filter(self.rad, 1.5, self.FS_Radar)

        self.wrapped_raw_phase = np.arctan2(
            self.radarWindowedSamples["I"], self.radarWindowedSamples["Q"]
        )
        self.unwrapped_raw_phase = np.unwrap(self.wrapped_raw_phase) * Radian2Meter

        self.br = self.filter_fun.butter_lowpass_filter(
            self.unwrapped_raw_phase, 0.2, self.FS_Radar
        )

        self.respBeats, _ = scipy.signal.find_peaks(
            self.br, distance=2 * self.FS_Radar
        )  # prominence=0.05)

        self.respBeats = np.array(self.respBeats)

        # self.predictBeats()

    def invalidatePrediction(self):
        self.predictedBeats = np.zeros(self.FS_Radar * self.window_size)
        self.beatpeaks = []

    def getHs(self):
        return self.hs

    def getPw(self):
        return self.pw

    def getBeats(self):
        return self.predictedBeats

    def getBeatPeaks(self):
        return self.beatpeaks

    def getBr(self):
        return self.br

    def getRespBeats(self):
        return self.respBeats
