import shutil
from pathlib import Path
from typing import Optional
import pandas as pd

from empkins_micro.feature_extraction.digital_biomarkers.acoustic.glottal_noise import calc_gne, empty_gne
from empkins_micro.feature_extraction.digital_biomarkers.acoustic.helper import process_segment_pitch
from empkins_micro.feature_extraction.digital_biomarkers.acoustic.jitter import calc_jitter, empty_jitter
from empkins_micro.feature_extraction.digital_biomarkers.acoustic.pause_segment import (
    calc_pause_segment,
    empty_pause_segment,
)
from empkins_micro.feature_extraction.digital_biomarkers.acoustic.shimmer import calc_shimmer, empty_shimmer
from empkins_micro.feature_extraction.digital_biomarkers.acoustic.voice_frame_score import calc_vfs, empty_vfs
from empkins_micro.feature_extraction.digital_biomarkers.movement.eyeblink import binarize_eyeblink
from empkins_micro.feature_extraction.digital_biomarkers.movement.voice_tremor import (
    calc_voicetremor,
    empty_voicetremor,
)
from empkins_micro.feature_extraction.digital_biomarkers.utils.segment_audio import segment_audio
from empkins_micro.utils._types import path_t


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
        self._do_feature_extraction = (
            self._df_pitch is not None or self._df_eyeblink is not None or self._diarization is not None
        )

    def _acoustic_fe(self):
        voice_seg, error_text = process_segment_pitch(self._df_pitch)

        if voice_seg is None:
            df_jitter = empty_jitter(error_text)
            df_shimmer = empty_shimmer(error_text)
            df_gne = empty_gne(error_text)
        else:
            df_jitter = calc_jitter(self._audio_path, voice_seg)
            df_shimmer = calc_shimmer(self._audio_path, voice_seg)
            df_gne = calc_gne(self._audio_path, voice_seg)

        return df_jitter, df_shimmer, df_gne

    def _eyeblink_fe(self):
        df_eyeblink = binarize_eyeblink(self._df_eyeblink)
        return df_eyeblink

    def _segmented_fe(self):
        path_files = self._audio_path.parent.joinpath("audio_segments")

        df_dia_seg, error_text = segment_audio(
            self._base_path, self._audio_path, self._subject_id, self._condition, self._diarization, path_files
        )

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
                tmp_audio_path_mono = tmp_audio_path.with_name(
                    f"{self._subject_id}_{self._condition}_seg_{idx}_mono"
                ).with_suffix(".wav")

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
                        "error": res_pause.at[0, "error"],
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
                        "error": res_vfs.at[0, "error"],
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
                        "error": res_vt.at[0, "error"],
                    }
                )

            df_pause_segment = pd.DataFrame(pause_segment)
            df_vfs = pd.DataFrame(vfs)
            df_vt = pd.DataFrame(vt)

        shutil.rmtree(path_files)

        return df_pause_segment, df_vfs, df_vt

    def extract_features(self):

        # jitter, shimmer, gne
        if self._df_pitch is not None:
            df_jitter, df_shimmer, df_gne = self._acoustic_fe()
        else:
            print("pitch dataframe (df_pitch) is not initialized")

        # eyeblink
        if self._df_eyeblink is not None:
            df_eyeblink = self._eyeblink_fe()
        else:
            print("eyeblink dataframe is not initialized")

        # segmented audio features
        if self._diarization is not None:
            df_pause_segment, df_vfs, df_vt = self._segmented_fe()
        else:
            print("speaker diarization is not initialized")

        df_dict = {
            "df_jitter": df_jitter,
            "df_shimmer": df_shimmer,
            "df_gne": df_gne,
            "df_eyeblink": df_eyeblink,
            "df_pause_segment": df_pause_segment,
            "df_vfs": df_vfs,
            "df_vt": df_vt,
        }

        return df_dict
