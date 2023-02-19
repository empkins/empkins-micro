import pandas as pd
import numpy as np
from pathlib import Path
from empkins_micro.utils._types import path_t
from typing import Optional
from empkins_micro.feature_extraction.movement.facial_tremor import calc_facial_tremor
from empkins_micro.feature_extraction.utils.derived_fe_dict import get_derived_fe_dict, get_fe_dict_structure
from empkins_micro.feature_extraction.derived.derived_features import var_range

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
            acoustic_seg_data: Optional[pd.DataFrame] = None,
            audio_seg_data: Optional[pd.DataFrame] = None,
            facial_tremor_data: Optional[pd.DataFrame] = None,
    ):

        self._base_path = base_path
        self._subject_id = subject_id
        self._condition = condition
        self._phase = phase
        self._facial_data = facial_data
        self._acoustic_data = acoustic_data
        self._movement_data = movement_data
        self._acoustic_seg_data = acoustic_seg_data
        self._audio_seg_data = audio_seg_data
        self._facial_tremor_data = facial_tremor_data

        if len(self._phase) > 1:
            print("features are derived from data of the whole video")

        self._feature_groups = get_fe_dict_structure()

    def calc_features(self, df, group, subgroup, fun_dict):
        fea_dict = get_derived_fe_dict(group=group, subgroup=subgroup)
        variables = list(fea_dict["raw_features"])
        features = list(fea_dict["derived_features"])
        df_derived = df[variables].agg(list(map(fun_dict.get, features)))
        df_derived = pd.DataFrame(df_derived.unstack()).T
        df_derived.columns = ["_".join(col) for col in df_derived.columns]
        return df_derived

    def derived_feature_extraction(self):

        fun_dict = {
            "mean": np.mean,
            "std": np.std,
            "range": lambda x: (np.max(x, axis=0) - np.min(x, axis=0)),
            "pct": ""
        }

        df = pd.DataFrame()
        if self._facial_data is not None:
            pass
            # df_facial_g1 = self.calc_features(
            #     df=self._facial_data, group="facial", subgroup="group_1", fun_dict=fun_dict)

        if self._acoustic_data is not None:
            for i in range(self._feature_groups["acoustic"]):
                tmp = self.calc_features(
                    df=self._acoustic_data, group="acoustic", subgroup=f"group_{i+1}", fun_dict=fun_dict)
                df = pd.concat([df, tmp], axis=1)

        if self._movement_data is not None:
            pass

        if self._acoustic_seg_data is not None:
            pass
            # df_aco_seg_g1 = self.calc_features(df=self._acoustic_seg_data, group="acoustic_seg", subgroup="group_1", fun_dict=fun_dict)

        if self._audio_seg_data is not None:
            pass

        return df

        # if self._facial_tremor_data is not None:
        #     df_facial_tremor = calc_facial_tremor(self._facial_tremor_data)
        #
        # return df_facial_tremor

