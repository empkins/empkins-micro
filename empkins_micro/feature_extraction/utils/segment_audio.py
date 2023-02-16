import pandas as pd
from pathlib import Path
import numpy as np
import os
import shutil
from pydub import AudioSegment
import math
import logging
from empkins_micro.utils._types import path_t


def _clean_diarization(diarization):
    dia = diarization.iloc[np.where(diarization["speaker"] == "SPEAKER_PANEL_INV")[0]]
    dia = dia.reset_index(drop=True)
    return dia


def _identify_test_subject(diarization):
    panel_inv = "SPEAKER_PANEL_INV"
    diarization = diarization[diarization.speaker != panel_inv].reset_index(drop=True)
    diarization = diarization.set_index("speaker")
    data = diarization[["length"]].groupby("speaker").sum()
    return data.idxmax()[0]


def _fix_stop_time(diarization):
    test_subject = _identify_test_subject(diarization)
    last_element = diarization[diarization.speaker == test_subject].tail(1)
    return np.float(last_element["stop"])


def segment_audio(base_path: path_t, audio_path: Path, subject_id: str, condition: str, diarization: pd.DataFrame,
                  files_path: Path):
    try:
        dia_segments = _clean_diarization(diarization=diarization)

        if files_path.exists():
            shutil.rmtree(files_path)
            os.mkdir(files_path)
        else:
            os.mkdir(files_path)

        logging.basicConfig(filename=base_path.joinpath("segment_audio_error.log"), filemode='a')

        with open(audio_path, "rb") as audio_file:
            audio = AudioSegment.from_file(audio_file, format="wav")

            for idx, seg in dia_segments.iterrows():
                tmp_path = files_path.joinpath(f"{subject_id}_{condition}_seg_{idx}.wav")
                t_start = seg['start'] * 1000
                t_stop = seg['stop'] * 1000

                if math.isnan(t_stop):
                    t_stop = _fix_stop_time(diarization) * 1000
                    dia_segments.loc[idx, "stop"] = t_stop / 1000
                    dia_segments.loc[idx, "length"] = dia_segments.loc[idx, "stop"] - dia_segments.loc[idx, "start"]
                    logging.warning(f"subject: {subject_id}, condition: {condition}, SPEAKER_PANEL_INV index: {idx},"
                                    f"SPEAKER_PANEL_INV max index {dia_segments.index[-1]}, "
                                    f"segment start time: {t_start/1000}")

                audio_segment = audio[t_start: t_stop]
                out = audio_segment.export(str(tmp_path), format="wav")
                out.close()

        logging.shutdown()
        return dia_segments, None
    except Exception as e:
        error_text = f"segmenting the audio failed: {e}"
        return None, error_text

