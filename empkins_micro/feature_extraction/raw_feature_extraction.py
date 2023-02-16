import pandas as pd
from pathlib import Path
from typing import Optional
import tarfile
import os
import shutil

from empkins_micro.utils._types import path_t
from empkins_micro.feature_extraction.acoustic.shimmer import calc_shimmer, empty_shimmer
from empkins_micro.feature_extraction.acoustic.jitter import calc_jitter, empty_jitter
from empkins_micro.feature_extraction.acoustic.glottal_noise import calc_gne, empty_gne
from empkins_micro.feature_extraction.acoustic.pause_segment import calc_pause_segment, empty_pause_segment
from empkins_micro.feature_extraction.movement.voice_tremor import calc_voicetremor, empty_voicetremor
from empkins_micro.feature_extraction.acoustic.voice_frame_score import calc_vfs, empty_vfs
from empkins_micro.feature_extraction.movement.eyeblink import binarize_eyeblink
from empkins_io.datasets.d03.macro_prestudy.helper import build_opendbm_tarfile_path, build_opendbm_raw_data_path
from empkins_micro.feature_extraction.acoustic.helper import process_segment_pitch
from empkins_micro.feature_extraction.utils import segment_audio


class RawFeatureExtraction:
    _base_path: path_t
    _subject_id: str
    _condition: str
    _audio_path: path_t
    _diarization: pd.DataFrame
    _df_pitch: pd.DataFrame
    _df_eyeblink: pd.DataFrame

    def __init__(
            self,
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

    def _get_data_path(self):
        return self._base_path.joinpath("data_per_subject", f"{self._subject_id}", f"{self._condition}", "video",
                                        "processed", "output")

    def _extract_opendbm_data(self):
        tarfile_path = build_opendbm_tarfile_path(base_path=self._base_path.joinpath("data_per_subject"),
                                                  subject_id=self._subject_id, condition=self._condition)
        data_path = self._get_data_path().parent

        with tarfile.open(tarfile_path, "r:gz") as tar:
            tar.extractall(path=data_path)

    def _compress_opendbm_data(self):

        tarfile_path_new = build_opendbm_tarfile_path(base_path=self._base_path.joinpath("data_per_subject"),
                                                      subject_id=self._subject_id, condition=self._condition,
                                                      new=True)

        data_path = self._get_data_path()

        with tarfile.open(tarfile_path_new, "w:gz") as tar:
            tar.add(data_path, arcname=os.path.basename(data_path))

    def write_file(self, df, group, subgroup, data_path):
        path = build_opendbm_raw_data_path(
            subject_id=self._subject_id, condition=self._condition, group=group, subgroup=subgroup
        )[0]
        path = data_path.parent.joinpath(path)
        os.mkdir(path.parent)
        df.to_csv(path, index=False)

    def _acoustic_fe(self, data_path):
        voice_seg, error_text = process_segment_pitch(self._df_pitch)

        if voice_seg is None:
            df_jitter = empty_jitter(error_text)
            df_shimmer = empty_shimmer(error_text)
            df_gne = empty_gne(error_text)
        else:
            df_jitter = calc_jitter(self._audio_path, voice_seg)
            df_shimmer = calc_shimmer(self._audio_path, voice_seg)
            df_gne = calc_gne(self._audio_path, voice_seg)

        self.write_file(df_jitter, "acoustic", "jitter_recomputed", data_path)
        self.write_file(df_shimmer, "acoustic", "shimmer_recomputed", data_path)
        self.write_file(df_gne, "acoustic", "gne_recomputed", data_path)

    def _eyeblink_fe(self, data_path):
        df_eyeblink = binarize_eyeblink(self._df_eyeblink)
        self.write_file(df_eyeblink, "movement", "eyeblink_binarized", data_path)

    def _segmented_fe(self, data_path):
        path_files = self._audio_path.parent.joinpath("audio_segments")

        df_dia_seg, error_text = segment_audio(self._base_path, self._audio_path, self._subject_id,
                                               self._condition, self._diarization, path_files)

        if df_dia_seg is None:
            df_pause_segment = empty_pause_segment(error_text)
            df_vfs = empty_vfs(error_text)
            df_vt = empty_voicetremor(error_text)
        else:
            pause_segment = []
            vfs = []
            vt = []

            for idx, seg in df_dia_seg.iterrows():
                tmp_audio_path = path_files.joinpath(f"{self._subject_id}_{self._condition}_seg_{idx}.wav")
                tmp_audio_path_mono = tmp_audio_path.with_name(f"{self._subject_id}_{self._condition}_seg_{idx}_mono") \
                    .with_suffix('.wav')

                # pause segment
                res_pause = calc_pause_segment(tmp_audio_path, tmp_audio_path_mono)
                pause_segment.append(
                    {
                        "start": seg["start"],
                        "stop": seg["stop"],
                        "length": seg["length"],
                        "aco_totaltime": res_pause.at[0, "aco_totaltime"],
                        "aco_speakingtime": res_pause.at[0, "aco_speakingtime"],
                        "aco_numpauses": res_pause.at[0, "aco_numpauses"],
                        "aco_pausetime": res_pause.at[0, "aco_pausetime"],
                        "aco_pausefrac": res_pause.at[0, "aco_pausefrac"],
                        "error": res_pause.at[0, "error"]
                    }
                )

                # voice frame score
                res_vfs = calc_vfs(tmp_audio_path)
                vfs.append(
                    {
                        "start": seg["start"],
                        "stop": seg["stop"],
                        "length": seg["length"],
                        "aco_voicepct": res_vfs.at[0, "aco_voicepct"],
                        "error": res_vfs.at[0, "error"]
                    }
                )

                # voice tremor
                res_vt = calc_voicetremor(tmp_audio_path)
                vt.append(
                    {
                        "start": seg["start"],
                        "stop": seg["stop"],
                        "length": seg["length"],
                        "mov_freqtremfreq": res_vt.at[0, "mov_freqtremfreq"],
                        "mov_amptremfreq": res_vt.at[0, "mov_amptremfreq"],
                        "mov_freqtremindex": res_vt.at[0, "mov_freqtremindex"],
                        "mov_amptremindex": res_vt.at[0, "mov_amptremindex"],
                        "mov_freqtrempindex": res_vt.at[0, "mov_freqtrempindex"],
                        "mov_amptrempindex": res_vt.at[0, "mov_amptrempindex"],
                        "error": res_vt.at[0, "error"]
                    }
                )

            df_pause_segment = pd.DataFrame(pause_segment)
            df_vfs = pd.DataFrame(vfs)
            df_vt = pd.DataFrame(vt)

        self.write_file(df_pause_segment, "acoustic", "pause_segment_recomputed", data_path)
        self.write_file(df_vfs, "acoustic", "voice_frame_score_recomputed", data_path)
        self.write_file(df_vt, "movement", "voice_tremor_recomputed", data_path)

        shutil.rmtree(path_files)

    def feature_extraction(self):
        data_path = self._get_data_path()

        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        if self._do_feature_extraction:
            self._extract_opendbm_data()

        # jitter, shimmer, gne
        if self._df_pitch is not None:
            self._acoustic_fe(data_path)
        else:
            print('pitch dataframe (df_pitch) is not initialized')

        # eyeblink
        if self._df_eyeblink is not None:
            self._eyeblink_fe(data_path)
        else:
            print('eyeblink dataframe is not initialized')

        # segmented audio features
        if self._diarization is not None:
            self._segmented_fe(data_path)
        else:
            print("speaker diarization is not initialized")

        if self._do_feature_extraction:
            self._compress_opendbm_data()

        shutil.rmtree(data_path)
