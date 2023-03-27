from typing import Dict, Optional

import numpy as np
import pandas as pd

import empkins_micro.feature_extraction.digital_biomarkers.derived_features._derived_features as derived_features
from empkins_micro.feature_extraction.digital_biomarkers.derived_features.pause_segment import pause_features
from empkins_micro.feature_extraction.digital_biomarkers.derived_features.utils import (
    get_derived_fe_dict,
    get_fe_dict_structure,
    get_subgroup
)
from empkins_micro.feature_extraction.digital_biomarkers.movement.facial_tremor import calc_facial_tremor
from empkins_micro.feature_extraction.digital_biomarkers.utils import DBM_FEATURE_GROUPS
from empkins_micro.utils._types import path_t


class DerivedFeatureExtraction:
    _base_path: path_t
    _subject_id: str
    _condition: str
    _phase: str
    _feature_data_dict: dict
    _feature_groups: dict

    def __init__(
        self, base_path: path_t, subject_id: str, condition: str, phase: str, feature_data_dict: Dict[str, pd.DataFrame]
    ):

        self._base_path = base_path
        self._subject_id = subject_id
        self._condition = condition
        self._phase = phase
        self._feature_data_dict = feature_data_dict.copy()

        for feature_group in DBM_FEATURE_GROUPS:
            # make sure that all feature groups are present in the feature_data_dict
            if feature_group not in self._feature_data_dict or len(self._feature_data_dict.get(feature_group, [])) == 0:
                self._feature_data_dict[feature_group] = None

        if len(self._phase) > 1:
            print("features are derived from data of the whole video")

        self._feature_groups = get_fe_dict_structure()

    def _calc_features(self, df, data, group, weights: Optional[np.array] = None):
        mean_weighted = lambda x: derived_features.mean_weighted(x, weights)
        mean_weighted.__name__ = "wmean"

        std_weighted = lambda x: derived_features.std_weighted(x, weights)
        std_weighted.__name__ = "wstd"

        fun_dict = {
            "mean": derived_features.mean,
            "std": derived_features.std,
            "wmean": mean_weighted,
            "wstd": std_weighted,
            "range": derived_features.range,
            "pct": derived_features.pct,
            "count": derived_features.count,
            "dur_mean": derived_features.dur_mean,
            "dur_std": derived_features.dur_std,
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
            names = ["_".join([v, f]) for v in variables for f in features]
            tmp = pd.DataFrame(data=np.zeros((1, len(names))).fill(np.nan), columns=names)
            df = pd.concat([df, tmp], axis=1)
        return df

    def extract_features(self):
        df = pd.DataFrame()

        for feature_group in DBM_FEATURE_GROUPS:
            # skip audio_seg and facial_tremor, they are calculated in the next step
            if feature_group in ["acoustic_seg", "audio_seg", "facial_tremor"]:
                continue
            data = self._feature_data_dict.get(feature_group, None)
            df = (
                self._calc_features(df, data, feature_group)
                if data is not None
                else self._empty_features(df, feature_group)
            )

        acoustic_seg = self._feature_data_dict.get("acoustic_seg", None)
        if acoustic_seg is not None:  # acoustic segmented data
            weights = (acoustic_seg["end_time"] - acoustic_seg["start_time"]).to_numpy()
            df = self._calc_features(df, acoustic_seg, "acoustic_seg", weights)
        else:
            df = self._empty_features(df, "acoustic_seg")

        audio_seg = self._feature_data_dict.get("audio_seg", None)
        if audio_seg is not None:
            weights = audio_seg["length"].to_numpy()
            df = self._calc_features(df, audio_seg, "audio_seg", weights)
            tmp = pause_features(audio_seg)  # pause features
            df = pd.concat([df, tmp], axis=1)
        else:
            df = self._empty_features(df, "audio_seg")
            df = self._empty_features(df, "pause")

        facial_tremor = self._feature_data_dict.get("facial_tremor", None)
        if facial_tremor is not None:
            tmp = calc_facial_tremor(facial_tremor)
            tmp = tmp.drop(["error"], axis=1)
            df = pd.concat([df, tmp], axis=1)
        else:
            df = self._empty_features(df, "facial_tremor")

        # concatenate middle parts of feature names that consist of more than three parts
        df.columns = [
            '_'.join([c.split('_')[0], ''.join(c.split('_')[1:-1]), c.split('_')[-1]]) if len(
                c.split('_')) > 3 else c
            for c in list(df.columns)]

        # create long formate dataframe
        df_long = []
        for col in list(df.columns):
            tmp = [self._subject_id, self._condition, self._phase[0]]
            col_split = list(col.split('_'))
            subgroup = get_subgroup(col_split[0], col_split[1])
            tmp.extend([col_split[0], subgroup, col_split[1], col_split[2], df.at[0, col]])
            df_long.append(tmp)

        df_long = pd.DataFrame(df_long,
                               columns=["subject", "condition", "phase", "group", "subgroup", "feature", "metric",
                                        "data"])

        # in opendbm documentation facial tremor belongs to movement group but is labeled with "fac" for facial
        # for correctness the label is changed to "mov"
        df_long.loc[np.where(df_long["subgroup"] == "facial_tremor")[0], "group"] = "mov"

        return df, df_long
