import pandas as pd
import numpy as np
from empkins_micro.utils._types import path_t
from typing import Optional
from empkins_micro.feature_extraction.movement.facial_tremor import calc_facial_tremor
from empkins_micro.feature_extraction.utils.derived_fe_dict import get_derived_fe_dict, get_fe_dict_structure


class DerivedFeatureExtraction:
    _base_path: path_t
    _subject_id: str
    _condition: str
    _phase: str
    _facial_data: pd.DataFrame
    _acoustic_data: pd.DataFrame
    _movement_data: pd.DataFrame
    _acoustic_seg_data: pd.DataFrame
    _audio_seg_data: pd.DataFrame
    _facial_tremor_data: pd.DataFrame
    _eyeblink_ear_data: pd.DataFrame
    _feature_groups: dict

    def __init__(
            self,
            base_path: path_t,
            subject_id: str,
            condition: str,
            phase: str,
            facial_data: Optional[pd.DataFrame] = None,
            acoustic_data: Optional[pd.DataFrame] = None,
            movement_data: Optional[pd.DataFrame] = None,
            eyeblink_ear_data: Optional[pd.DataFrame] = None,
            acoustic_seg_data: Optional[pd.DataFrame] = None,
            audio_seg_data: Optional[pd.DataFrame] = None,
            facial_tremor_data: Optional[pd.DataFrame] = None
    ):

        self._base_path = base_path
        self._subject_id = subject_id
        self._condition = condition
        self._phase = phase
        self._facial_data = facial_data
        self._acoustic_data = acoustic_data
        self._movement_data = movement_data
        self._eyeblink_ear_data = eyeblink_ear_data
        self._acoustic_seg_data = acoustic_seg_data
        self._audio_seg_data = audio_seg_data
        self._facial_tremor_data = facial_tremor_data

        if len(self._phase) > 1:
            print("features are derived from data of the whole video")

        # sanity check
        if self._facial_data is not None and len(self._facial_data.index) == 0:
            print("facial_data is not available and set to None")
            self._facial_data = None

        if self._acoustic_data is not None and len(self._acoustic_data.index) == 0:
            print("acoustic_data is not available and set to None")
            self._acoustic_data = None

        if self._movement_data is not None and len(self._movement_data.index) == 0:
            print("movement_data is not available and set to None")
            self._movement_data = None

        if self._eyeblink_ear_data is not None and len(self._eyeblink_ear_data.index) == 0:
            print("eyeblink_ear_data is not available and set to None")
            self._eyeblink_ear_data = None

        if self._acoustic_seg_data is not None and len(self._acoustic_seg_data.index) == 0:
            print("acoustic_seg_data is not available and set to None")
            self._acoustic_seg_data = None

        if self._audio_seg_data is not None and len(self._audio_seg_data.index) == 0:
            print("audio_seg_data is not available and set to None")
            self._audio_seg_data = None

        if self._facial_tremor_data is not None and len(self._facial_tremor_data.index) == 0:
            print("facial_tremor_data is not available and set to None")
            self._facial_tremor_data = None

        self._feature_groups = get_fe_dict_structure()

    def _get_weights(self, weight_type):
        if self._acoustic_seg_data is not None:
            num_values_aco = len(self._acoustic_seg_data.index)
        else:
            num_values_aco = None

        if self._audio_seg_data is not None:
            num_values_audio = len(self._audio_seg_data.index)
        else:
            num_values_audio = None

        if weight_type == num_values_aco:
            weights = (self._acoustic_seg_data["end_time"] - self._acoustic_seg_data["start_time"]).to_numpy()
        elif weight_type == num_values_audio:
            weights = self._audio_seg_data["length"].to_numpy()
        else:
            weights = None
        return weights

    def mean(self, data):
        df = data.astype(float).copy()
        df = df.dropna().reset_index(drop=True)
        val = df.mean(axis=0, skipna=True)
        return val

    def std(self, data):
        df = data.astype(float).copy()
        df = df.dropna().reset_index(drop=True)
        val = df.std(axis=0, skipna=True)
        return val

    def range(self, data):
        df = data.astype(float).copy()
        df = df.dropna().reset_index(drop=True)
        val = max(df) - min(df)
        return val

    def pct(self, data):
        df = data.astype(float).copy()
        df = df.dropna().reset_index(drop=True)
        val = len(df[df > 0.0]) / len(df)
        return val

    def wmean(self, data):
        weights = self._get_weights(len(data.index))
        if weights is not None:
            indices = data.isna()
            df = data.astype(float).copy()
            df = df.dropna().reset_index(drop=True)
            weights = weights[~indices]
            val = np.average(df, weights=weights)
            return val
        return np.nan

    def wstd(self, data):
        weights = self._get_weights(len(data.index))
        if weights is not None:
            indices = data.isna()
            df = data.astype(float).copy()
            df = df.dropna().reset_index(drop=True)
            weights = weights[~indices]
            val = np.sqrt(np.cov(df.to_numpy(), aweights=weights))
            return val
        return np.nan

    def count(self, data):
        val = sum(data) * 60.0 / data.index[-1]
        return val

    def _calc_blinks_diff(self, data):
        time = data.index.to_numpy()
        indices = np.where(data == 1)[0]
        time_blinks = time[indices]
        time_blinks_diff = np.diff(time_blinks)
        return time_blinks_diff

    def dur_mean(self, data):
        time_blinks_diff = self._calc_blinks_diff(data)
        val = np.mean(time_blinks_diff)
        return val

    def dur_std(self, data):
        time_blinks_diff = self._calc_blinks_diff(data)
        val = np.std(time_blinks_diff)
        return val

    def pause_features(self, data):
        features = {
            "aco_pausetime_mean": [data["aco_pausetime"].sum()],
            "aco_totaltime_mean": [data["aco_totaltime"].sum()],
            "aco_numpauses_mean": [data["aco_numpauses"].sum()],
            "aco_pausefrac_mean": [data["aco_pausetime"].sum() / data["aco_totaltime"].sum()]
        }
        return pd.DataFrame.from_dict(features)

    def _calc_features(self, df, group, subgroup, fun_dict):
        fea_dict = get_derived_fe_dict(group=group, subgroup=subgroup)
        variables = list(fea_dict["raw_features"])
        features = list(fea_dict["derived_features"])
        df_derived = df[variables].agg(list(map(fun_dict.get, features)))
        df_derived = pd.DataFrame(df_derived.unstack()).T
        df_derived.columns = ["_".join(col) for col in df_derived.columns]
        return df_derived

    def _run_calc_features(self, df, data, group):
        fun_dict = {
            "mean": self.mean,
            "std": self.std,
            "wmean": self.wmean,
            "wstd": self.wstd,
            "range": self.range,
            "pct": self.pct,
            "count": self.count,
            "dur_mean": self.dur_mean,
            "dur_std": self.dur_std
        }

        for i in range(self._feature_groups[group]):
            tmp = self._calc_features(
                df=data, group=group, subgroup=f"group_{i + 1}", fun_dict=fun_dict)
            df = pd.concat([df, tmp], axis=1)

        return df

    def _run_empty_features(self, df, group):
        for i in range(self._feature_groups[group]):
            fea_dict = get_derived_fe_dict(group=group, subgroup=f"group_{i + 1}")
            variables = list(fea_dict["raw_features"])
            features = list(fea_dict["derived_features"])
            names = ['_'.join([v, f]) for v in variables for f in features]
            tmp = pd.DataFrame(data=np.zeros((1, len(names))).fill(np.nan), columns=names)
            df = pd.concat([df, tmp], axis=1)
        return df

    def derived_feature_extraction(self):

        df = pd.DataFrame()

        if self._facial_data is not None:  # facial data
            df = self._run_calc_features(df, self._facial_data, "facial")
        else:
            df = self._run_empty_features(df, "facial")

        if self._acoustic_data is not None:  # acoustic data
            df = self._run_calc_features(df, self._acoustic_data, "acoustic")
        else:
            df = self._run_empty_features(df, "acoustic")

        if self._movement_data is not None:  # movement data
            df = self._run_calc_features(df, self._movement_data, "movement")
        else:
            df = self._run_empty_features(df, "movement")

        if self._eyeblink_ear_data is not None:  # movement - eyeblink - ear
            df = self._run_calc_features(df, self._eyeblink_ear_data, "eyeblink_ear")
        else:
            df = self._run_empty_features(df, "eyeblink_ear")

        if self._acoustic_seg_data is not None:  # acoustic segmented data
            df = self._run_calc_features(df, self._acoustic_seg_data, "acoustic_seg")
        else:
            df = self._run_empty_features(df, "acoustic_seg")

        if self._audio_seg_data is not None:  # audio segmented data
            df = self._run_calc_features(df, self._audio_seg_data, "audio_seg")
            tmp = self.pause_features(self._audio_seg_data)  # pause features
            df = pd.concat([df, tmp], axis=1)
        else:
            df = self._run_empty_features(df, "audio_seg")
            df = self._run_empty_features(df, "pause")

        if self._facial_tremor_data is not None:  # movement - facial tremor
            tmp = calc_facial_tremor(self._facial_tremor_data)
            tmp = tmp.drop(["error"], axis=1)
            df = pd.concat([df, tmp], axis=1)
        else:
            df = self._run_empty_features(df, "facial_tremor", )

        return df

