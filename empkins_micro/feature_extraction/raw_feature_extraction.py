import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import tarfile
import io
import os
import shutil

from empkins_micro.utils._types import path_t
from empkins_micro.feature_extraction.acoustic.shimmer import calc_shimmer
from empkins_micro.feature_extraction.acoustic.jitter import calc_jitter
from empkins_micro.feature_extraction.acoustic.glottal_noise import calc_gne
from empkins_micro.feature_extraction.movement.eyeblink import binarize_eyeblink
from empkins_io.datasets.d03.macro_prestudy.helper import build_opendbm_tarfile_path, build_opendbm_raw_data_path


class RawFeatureExtraction():
    _base_path: path_t
    _subject_id: str
    _condition: str
    _df_pitch: pd.DataFrame
    _audio_path: path_t
    _df_eyeblink: pd.DataFrame

    def __init__(self,
                 base_path: path_t,
                 subject_id: str,
                 condition: str,
                 df_pitch: Optional[pd.DataFrame] = None,
                 audio_path: Optional[path_t] = "",
                 df_eyeblink: Optional[pd.DataFrame] = None):

        self._base_path = base_path
        self._subject_id = subject_id
        self._condition = condition
        self._df_pitch = df_pitch
        self._audio_path = Path(audio_path)
        self._df_eyeblink = df_eyeblink

    def get_temporary_path(self):
        return self._base_path.joinpath("tmp_files")

    def write_file(self, base_path: path_t, subject_id: str, condition: str):

        tarfile_path = build_opendbm_tarfile_path(base_path=base_path, subject_id=subject_id, condition=condition)
        tarfile_path_new = build_opendbm_tarfile_path(base_path=base_path, subject_id=subject_id, condition=condition,
                                                      new=True)

        jitter_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                  group="acoustic", subgroup="jitter_recomputed")[0]
        shimmer_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                   group="acoustic", subgroup="shimmer_recomputed")[0]
        gne_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                               group="acoustic", subgroup="gne_recomputed")[0]
        eyeblink_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                    group="movement", subgroup="eyeblink_binarized")[0]

        tmp_path = self.get_temporary_path()

        with tarfile.open(tarfile_path, "r:gz") as tar:
            members = tar.getmembers()
            tar.extractall(path=tmp_path)

        with tarfile.open(tarfile_path_new, "w:gz") as tar:
            for member in members:
                tar.add(tmp_path.joinpath(member.name), arcname=member.name)

            if self._df_pitch is not None:
                tar.add(tmp_path.joinpath("jitter.csv"), arcname=jitter_path)
                tar.add(tmp_path.joinpath("shimmer.csv"), arcname=shimmer_path)
                tar.add(tmp_path.joinpath("gne.csv"), arcname=gne_path)

            if self._df_eyeblink is not None:
                tar.add(tmp_path.joinpath("eyeblink.csv"), arcname=eyeblink_path)

        # TODO
        # shutil.rmtree(tmp_path)

    def feature_extraction(self):

        tmp_path = self.get_temporary_path()
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        else:
            shutil.rmtree(tmp_path)
            os.mkdir(tmp_path)

        if self._df_pitch is not None:

            df_jitter = calc_jitter(self._df_pitch, self._audio_path)
            # df_shimmer = calc_shimmer(self._df_pitch, self._audio_path)
            # df_gne = calc_gne(self._df_pitch, self._audio_path)

            df_jitter.to_csv(tmp_path.joinpath("jitter.csv"), index=False)
            # df_shimmer.to_csv(tmp_path.joinpath("shimmer.csv"), index=False)
            # df_gne.to_csv(tmp_path.joinpath("gne.csv"), index=False)

        else:
            raise print('pitch dataframe (df_pitch) is not initialized')

        # if self._df_eyeblink is not None:
        #     df_eyeblink = binarize_eyeblink(self._df_eyeblink)
        #     df_eyeblink.to_csv(tmp_path.joinpath("eyeblink.csv"), index=False)
        #
        # else:
        #     raise print('eyeblink dataframe is not initialized')


        # if self._df_pitch is not None or self._df_eyeblink is not None:
        #     self.write_file(self._base_path.joinpath("data_per_subject"), self._subject_id, self._condition)

