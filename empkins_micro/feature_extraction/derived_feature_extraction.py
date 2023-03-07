import pandas as pd
import numpy as np
from empkins_micro.utils._types import path_t
from typing import Optional
from empkins_micro.feature_extraction.movement.facial_tremor import calc_facial_tremor
from empkins_micro.feature_extraction.utils.derived_fe_dict import get_derived_fe_dict, get_fe_dict_structure
import empkins_micro.feature_extraction.derived.derived_features as fea
from empkins_micro.feature_extraction.derived.pause_segment import pause_features


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

    def _calc_features(self, df, data, group, weights: Optional[np.array] = None):
        mean_weighted = lambda x: fea.mean_weighted(x, weights)
        mean_weighted.__name__ = "mean"

        std_weighted = lambda x: fea.std_weighted(x, weights)
        std_weighted.__name__ = "std"

        fun_dict = {
            "mean": fea.mean,
            "std": fea.std,
            "wmean": mean_weighted,
            "wstd": std_weighted,
            "range": fea.range,
            "pct": fea.pct,
            "count": fea.count,
            "dur_mean": fea.dur_mean,
            "dur_std": fea.dur_std
        }

        for i in range(self._feature_groups[group]):
            subgroup = f"group_{i + 1}"
            fea_dict = get_derived_fe_dict(group=group, subgroup=subgroup)
            variables = list(fea_dict["raw_features"])
            features = list(fea_dict["derived_features"])
            df_derived = data[variables].agg(list(map(fun_dict.get, features)))
            df_derived = pd.DataFrame(df_derived.unstack()).T
            df_derived.columns = ["_".join(col) for col in df_derived.columns]
            df = pd.concat([df, df_derived], axis=1)

        return df

    def _empty_features(self, df, group):
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
            df = self._calc_features(df, self._facial_data, "facial")
        else:
            df = self._empty_features(df, "facial")

        if self._acoustic_data is not None:  # acoustic data
            df = self._calc_features(df, self._acoustic_data, "acoustic")
        else:
            df = self._empty_features(df, "acoustic")

        if self._movement_data is not None:  # movement data
            df = self._calc_features(df, self._movement_data, "movement")
        else:
            df = self._empty_features(df, "movement")

        if self._eyeblink_ear_data is not None:  # movement - eyeblink - ear
            df = self._calc_features(df, self._eyeblink_ear_data, "eyeblink_ear")
        else:
            df = self._empty_features(df, "eyeblink_ear")

        if self._acoustic_seg_data is not None:  # acoustic segmented data
            weights = (self._acoustic_seg_data["end_time"] - self._acoustic_seg_data["start_time"]).to_numpy()
            df = self._calc_features(df, self._acoustic_seg_data, "acoustic_seg", weights)
        else:
            df = self._empty_features(df, "acoustic_seg")

        if self._audio_seg_data is not None:  # audio segmented data
            weights = self._audio_seg_data["length"].to_numpy()
            df = self._calc_features(df, self._audio_seg_data, "audio_seg", weights)
            tmp = pause_features(self._audio_seg_data)  # pause features
            df = pd.concat([df, tmp], axis=1)
        else:
            df = self._empty_features(df, "audio_seg")
            df = self._empty_features(df, "pause")

        if self._facial_tremor_data is not None:  # movement - facial tremor
            tmp = calc_facial_tremor(self._facial_tremor_data)
            tmp = tmp.drop(["error"], axis=1)
            df = pd.concat([df, tmp], axis=1)
        else:
            df = self._empty_features(df, "facial_tremor", )

        return df
