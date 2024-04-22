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
import time

from empkins_micro.emrad.utils import input_conversion, correct_peaks

Radar_Freq_Hz = 61 * 1e9  # 122 GHz
C_Light_Air_mps = 299708516  # reduzierte Lichtgeschwindigkeit in bodennaher Luft
Radian2Meter = C_Light_Air_mps / (4 * np.pi * Radar_Freq_Hz)
TIMESTEPS = 400
MAXBPM = 130
Minimumperiod = 60 / MAXBPM
DETECTIONTHRESHOLD = 0.05


def get_rpeaks(
    radar_data_1: pd.DataFrame, radar_data_2: pd.DataFrame, radar_data_3: pd.DataFrame, radar_data_4: pd.DataFrame,
    fs_radar: float, window_size: int, threshold: float):
    #-> bp.utils.datatype_helper.RPeakDataFrame:

    print("----- get lstm rad 1 ------")
    start = time.time_ns()
    lstm_1 = get_lstm(radar_data_1,fs_radar,window_size)
    end = time.time_ns()
    print('time for get_lstm rad 1: ' + str((end - start) / (10 ** 9)) + ' s, in min: ' + str(
        ((end - start) / (10 ** 9)) / 60))
    print("----- get lstm rad 2 ------")
    lstm_2 = get_lstm(radar_data_2,fs_radar,window_size)
    print("----- get lstm rad 3 ------")
    lstm_3 = get_lstm(radar_data_3,fs_radar,window_size)
    print("----- get lstm rad 4 ------")
    lstm_4 = get_lstm(radar_data_4,fs_radar,window_size)

    print("------- concat lstm --------")
    start = time.time_ns()
    lstm_concat = pd.DataFrame({"predicted_beats_1": lstm_1["predicted_beats"],
                  "predicted_beats_2": lstm_2["predicted_beats"],
                  "predicted_beats_3": lstm_3["predicted_beats"],
                  "predicted_beats_4": lstm_4["predicted_beats"]})
    end = time.time_ns()
    print('time for concat: ' + str((end - start) / (10 ** 9)) + ' s, in min: ' + str(((end - start) / (10 ** 9)) / 60))

    print("-------- sum lstm ----------")
    start = time.time_ns()
    lstm_sum = pd.DataFrame({"predicted_beats": lstm_concat.sum(axis=1, min_count=1)})
    end = time.time_ns()
    print('time for sum: ' + str((end - start) / (10 ** 9)) + ' s, in min: ' + str(((end - start) / (10 ** 9)) / 60))

    print("------- find peaks of sum lstm --------")
    start = time.time_ns()
    radar_beats, beats_prop = get_pred_peaks(lstm_sum, fs_radar, threshold)
    end = time.time_ns()
    print('time for get_pred_peaks: ' + str((end - start) / (10 ** 9)) + ' s, in min: ' + str(((end - start) / (10 ** 9)) / 60))

    return radar_beats, lstm_sum

def get_pred_peaks(lstm_sum: pd.DataFrame, fs_radar: float, threshold: float
                   ):#-> bp.utils.datatype_helper.RPeakDataFrame:
    radar_beats,peak_prop = find_peaks(
        lstm_sum.predicted_beats, height=threshold, distance=0.3 * fs_radar, width=None, prominence=None
    )
    #radar_beats = find_peaks(
    #    lstm_sum.predicted_beats, height=threshold, distance=0.3 * fs_radar
    #)[0]
    radar_beats = pd.DataFrame(
        radar_beats, index=lstm_sum.index[radar_beats], columns=["peak_idx"]
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

    #bp.signals.ecg.EcgProcessor.correct_outlier(
    #    rpeaks=radar_beats,
    #    sampling_rate=fs_radar,
    #    imputation_type="moving_average",
    #    outlier_correction=["physiological", "statistical_rr", "statistical_rr_diff"],
    #)

    return radar_beats, peak_prop

def get_lstm(radar_data: pd.DataFrame, fs_radar: float, window_size: int):
    data_out = {}
    duration = (radar_data.index[-1] - radar_data.index[0]).total_seconds()
    num_windows = int(duration // window_size)
    print("------ duration ------")
    print(duration)
    print("------ num_windows before and after factoring overlap------")
    print(num_windows)
    # num_win for slinding window with overlap 50%
    overlap = 2
    num_windows = num_windows * overlap + 1
    if num_windows * window_size / overlap > duration:
        num_windows = num_windows - 1
    print(num_windows)
    cut_off = int((window_size / (overlap * 2) * fs_radar))


    processing = Processing(FS=fs_radar, window_size=window_size)

    for wind_ctr in range(num_windows):
        # shift the window by overlap of the window size
        start_sample_radar = int(np.ceil(wind_ctr * (window_size / overlap) * fs_radar))
        end_sample_radar = min(
            start_sample_radar + int(np.floor(fs_radar * window_size)), len(radar_data)
        )
       # print("window size" + str((radar_data.index[end_sample_radar]-radar_data.index[start_sample_radar]).total_seconds()))
        radar_slice = radar_data.iloc[start_sample_radar:end_sample_radar]

        processing.setRadarSamples(radar_slice)
        processing.predictBeats()

        predicted_beats = processing.getBeats()[["predicted_beats"]]

        if start_sample_radar + cut_off + 1 == min(int(np.ceil((wind_ctr-1) * (window_size / overlap) * fs_radar)) + int(np.floor(fs_radar * window_size)) - cut_off,
                                                len(radar_data)):
            data_out[wind_ctr] = predicted_beats[cut_off+2:-cut_off]
        data_out[wind_ctr] = predicted_beats[cut_off+1:-cut_off]

    if len(data_out) > 1:
        data_concat = pd.concat(data_out, names=["window_id"])
    else:
        data_concat = pd.DataFrame(predicted_beats)

    data_concat.index = data_concat.index.droplevel(0)
    return data_concat

def transform_for_nk_hrv(
        peaks: pd.DataFrame, lstm_output: pd.DataFrame, fs_radar: float
):
    print("------ transform_for_nk_hrv ------")
    print("------ extrakt peak position in peaks ------")
    drop_nan_peaks = peaks['R_Peak_Idx'].dropna()
    pos_in_time = np.empty_like(peaks['R_Peak_Idx'], dtype='object')
    peak_ind = np.empty_like(peaks['R_Peak_Idx'], dtype='int64')
    j = 0
    for i in peaks['R_Peak_Idx']:
        pos_in_time[j] = lstm_output.index[int(i)]
        peak_ind[j] = int(i)
        j = j + 1

    print("------ extrakt time index of peaks ------")
    time_to_int = np.arange(len(lstm_output), dtype='int64')
    val_peak = np.zeros_like(time_to_int, dtype='int64')
    j = 0
    for i in range(len(lstm_output)):
        if lstm_output.index[i] == pos_in_time[j]:
            val_peak[i] = 1
            if j != len(pos_in_time)-1:
                j = j + 1

    print("------ create dict and hrv_input------")
    info = {'method_peaks': 'emkins_micro',
            'method_fixpeaks': 'None',
            'ECG_R_Peaks': peak_ind,
            'sampling_rate': fs_radar}

    hrv_input_peak = pd.DataFrame({"time": lstm_output.index,
                                   "ECG_R_Peaks": val_peak
                                   },
                                    index=time_to_int)

    return hrv_input_peak, info

def get_hrv_featurs(window_size:int,window_step:int, hrv_input_peak:pd.DataFrame, fs_radar:float, overlap:int):

    print("Window_size in s: " + str(window_size))
    window_size = window_size
    window_step = window_step
    d = {}
    temp_val = {}

    duration = (hrv_input_peak["time"].iloc[-1] - hrv_input_peak["time"].iloc[0]).total_seconds()
    num_windows = int(duration // window_step)
    print("------ duration ------")
    print(f"duration of mesurment in sec: {duration}")
    print(f"duration of mesurment in min: {duration/60}")
    print(f"duration of mesurment in h: {duration/3600}")
    print("------ num_windows before and after factoring overlap------")
    print(num_windows)
        # num_win for slinding window with overlap 50%
    overlap = overlap
    num_windows = num_windows * overlap + 1
    if num_windows * window_size / overlap > duration:
        num_windows = num_windows - 1

    print(num_windows)
    print("----win_loop_start----")
    array_middel_sample_radar = np.zeros(num_windows, dtype=int)
    magic_minus = 1
    for wind_ctr in range(num_windows):
        #window stays as long as 30 sec. epoch(step size) is smaller as the half of the window size at the same value
        # just when epoch is biber than half the window size the window is shifted by 30sec of the step size
        array_middel_sample_radar[wind_ctr] = max(int(np.ceil((wind_ctr * (window_step // overlap) * fs_radar))),
                                                  int(np.ceil(((window_size // 2) * fs_radar))))
        start_sampel = array_middel_sample_radar[wind_ctr] - int(np.ceil(((window_size // 2) * fs_radar)))
        end_sampel = min(array_middel_sample_radar[wind_ctr] + int(np.ceil(((window_size//2)* fs_radar))), len(hrv_input_peak)-1)
        # Randbehandlung repeat last valid window til end of data / put 0 in the window
        if end_sampel == len(hrv_input_peak) - 1:
            d[wind_ctr] = d[wind_ctr - magic_minus]  # write the last full window in the following windows
            # d[wind_ctr] = pd.DataFrame(0, index=[wind_ctr], columns=d[wind_ctr-magic_minus].columns) # to fill with zeros
            #temp val showcases the time step of the epochs which will also be the time of the middle of the window after the middle is reached
            temp_val[wind_ctr] = pd.DataFrame(
                {"time": hrv_input_peak["time"].loc[int(np.ceil((wind_ctr * (window_step // overlap) * fs_radar)))]}, index=[wind_ctr])
            # temp_val[wind_ctr] = pd.DataFrame({"time": 0} index=[wind_ctr])
            magic_minus = magic_minus + 1
            continue

        if array_middel_sample_radar[wind_ctr-1] != start_sampel and array_middel_sample_radar[wind_ctr-1] != 0:
            peak_window = hrv_input_peak.iloc[start_sampel+1:end_sampel+1]
         #   print(
         #       f"start_sampel - end_sample, peak_wind_len, middel ad: {start_sampel+1, end_sampel+1, len(peak_window), array_middel_sample_radar[wind_ctr],}")
        else:
            peak_window = hrv_input_peak.iloc[start_sampel:end_sampel]
         #   print(
          #      f"start_sampel - end_sample, peak_wind_len, middel de: {start_sampel, end_sampel, len(peak_window), array_middel_sample_radar[wind_ctr],}")

        #print(f"start_sampel - end_sample, peak_wind_len : {start_sampel, end_sampel, len(peak_window)}")

        #hrv_window = nk.hrv(peak_window, sampling_rate=fs_radar)
        hrv_time = nk.hrv_time(peak_window, sampling_rate=fs_radar)
        hrv_frequ = nk.hrv_frequency(peak_window, sampling_rate=fs_radar)
        hrv_nonlin = nk.hrv_nonlinear(peak_window, sampling_rate=fs_radar)
        hrv_window = pd.concat([hrv_time, hrv_frequ, hrv_nonlin], axis=1)
        d[wind_ctr] = hrv_window
        temp_val[wind_ctr] = pd.DataFrame({"time": hrv_input_peak["time"].loc[int(np.ceil((wind_ctr * (window_step // overlap) * fs_radar)))]},
                                              index=[wind_ctr])
       # print(f"temp_val[wind_ctr], middel time: {temp_val[wind_ctr]}")

    print("-------- concat data  --------")
    temp = pd.concat(temp_val)
    temp.index = temp.index.droplevel(1)
    conc_d = pd.concat(d, names=["window_id"])
    conc_d.index = conc_d.index.droplevel(1)
    data_concat = pd.concat([temp, conc_d], axis=1, names=["window_id", "time"])
    data_concat.index.names = ['window_id']
    data_concat.dropna(axis=1,how='all',inplace=True)
    #data_concat = data_concat.replace(np.nan, 0)
    return data_concat


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
