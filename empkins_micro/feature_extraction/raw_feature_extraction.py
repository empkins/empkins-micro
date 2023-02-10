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
from empkins_micro.feature_extraction.acoustic.helper import process_segment_pitch
from empkins_micro.feature_extraction.acoustic.pause_segment import run_pause_segment
from empkins_micro.feature_extraction.acoustic.voice_frame_score import calc_vfs
from empkins_micro.feature_extraction.acoustic.voice_tremor import tremor_praat
from empkins_micro.feature_extraction.utils import segment_audio


class RawFeatureExtraction():
    _base_path: path_t
    _subject_id: str
    _condition: str
    _audio_path: path_t
    _diarization: pd.DataFrame
    _df_pitch: pd.DataFrame
    _df_eyeblink: pd.DataFrame
    _features_audio_seg: bool

    def __init__(self,
                 base_path: path_t,
                 subject_id: str,
                 condition: str,
                 audio_path: Optional[path_t] = "",
                 diarization: Optional[pd.DataFrame] = None,
                 df_pitch: Optional[pd.DataFrame] = None,
                 df_eyeblink: Optional[pd.DataFrame] = None,
                 ):

        self._base_path = base_path
        self._subject_id = subject_id
        self._condition = condition
        self._audio_path = Path(audio_path)
        self._diarization = diarization
        self._df_pitch = df_pitch
        self._df_eyeblink = df_eyeblink
        self._do_feature_extraction = self._df_pitch is not None or self._df_eyeblink is not None or \
                                      self._diarization is not None

    def get_data_path(self):
        return self._base_path.joinpath("data_per_subject", f"{self._subject_id}", f"{self._condition}", "video",
                                        "processed", "output")

    def extract_opendbm_data(self):
        tarfile_path = build_opendbm_tarfile_path(base_path=self._base_path.joinpath("data_per_subject"),
                                                  subject_id=self._subject_id, condition=self._condition)
        data_path = self.get_data_path().parent

        with tarfile.open(tarfile_path, "r:gz") as tar:
            tar.extractall(path=data_path)

    def compress_opendbm_data(self):

        tarfile_path_new = build_opendbm_tarfile_path(base_path=self._base_path.joinpath("data_per_subject"),
                                                      subject_id=self._subject_id, condition=self._condition,
                                                      new=True)

        data_path = self.get_data_path()

        with tarfile.open(tarfile_path_new, "w:gz") as tar:
            tar.add(data_path, arcname=os.path.basename(data_path))

    def feature_extraction(self):
        data_path = self.get_data_path()

        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        # if self._do_feature_extraction:
        #     self.extract_opendbm_data()

        # jitter, shimmer, gne
        if self._df_pitch is not None:
            voice_seg = process_segment_pitch(self._df_pitch)

            df_jitter = calc_jitter(self._audio_path, voice_seg)
            df_shimmer = calc_shimmer(self._audio_path, voice_seg)
            df_gne = calc_gne(self._audio_path, voice_seg)

            jitter_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                      group="acoustic", subgroup="jitter_recomputed")[0]
            jitter_path = data_path.parent.joinpath(jitter_path)
            os.mkdir(jitter_path.parent)
            shimmer_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                       group="acoustic", subgroup="shimmer_recomputed")[0]
            shimmer_path = data_path.parent.joinpath(shimmer_path)
            os.mkdir(shimmer_path.parent)
            gne_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                   group="acoustic", subgroup="gne_recomputed")[0]
            gne_path = data_path.parent.joinpath(gne_path)
            os.mkdir(gne_path.parent)

            df_jitter.to_csv(jitter_path, index=False)
            df_shimmer.to_csv(shimmer_path, index=False)
            df_gne.to_csv(gne_path, index=False)

        else:
            print('pitch dataframe (df_pitch) is not initialized')

        # eyeblink
        if self._df_eyeblink is not None:
            df_eyeblink = binarize_eyeblink(self._df_eyeblink)

            eyeblink_path = build_opendbm_raw_data_path(subject_id=self._subject_id, condition=self._condition,
                                                        group="movement", subgroup="eyeblink_binarized")[0]
            eyeblink_path = data_path.parent.joinpath(eyeblink_path)
            os.mkdir(eyeblink_path.parent)
            df_eyeblink.to_csv(eyeblink_path, index=False)

        else:
            print('eyeblink dataframe is not initialized')

        # compute features from segmented audio files
        if self._diarization is not None:
            path_files = self._audio_path.parent.joinpath("audio_segments")

            df_dia_seg = segment_audio(self._audio_path, self._subject_id,
                                            self._condition, self._diarization, path_files)

            # df_pause_segment = pd.DataFrame(columns=["start", "stop", "length", ])
            # for idx, seg in df_dia_seg:


            # audio_path_mono = self..with_name(f"{self._subject_id}_{self._condition}_mono").with_suffix('.wav')

            # run_pause_segment(self._audio_path, audio_path_mono)
            # calc_vfs(self._audio_path)
            # tremor_praat(self._audio_path)
            return #data


        else:
            print("features from segmented audio files are not computed")

        # if self._do_feature_extraction:
        #     self.compress_opendbm_data()

        # shutil.rmtree(data_path)
