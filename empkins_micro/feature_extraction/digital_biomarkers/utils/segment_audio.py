import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment

from empkins_micro.utils._types import path_t
from empkins_io.datasets.d03.macro_prestudy.helper import clean_diarization, fix_stop_time


def segment_audio(
    base_path: path_t, audio_path: Path, subject_id: str, condition: str, diarization: pd.DataFrame, files_path: Path
):
    try:
        dia_segments = clean_diarization(diarization=diarization)

        if files_path.exists():
            shutil.rmtree(files_path)
        files_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(filename=base_path.joinpath("segment_audio_error.log"), filemode="a")

        with open(audio_path, "rb") as audio_file:
            audio = AudioSegment.from_file(audio_file, format="wav")

            for idx, seg in dia_segments.iterrows():
                tmp_path = files_path.joinpath(f"{subject_id}_{condition}_seg_{idx}.wav")
                t_start = seg["start"] * 1000
                t_stop = seg["stop"] * 1000

                if math.isnan(t_stop):
                    t_stop = fix_stop_time(diarization) * 1000
                    dia_segments.loc[idx, "stop"] = t_stop / 1000
                    dia_segments.loc[idx, "length"] = dia_segments.loc[idx, "stop"] - dia_segments.loc[idx, "start"]
                    logging.warning(
                        f"subject: {subject_id}, condition: {condition}, SPEAKER_PANEL_INV index: {idx},"
                        f"SPEAKER_PANEL_INV max index {dia_segments.index[-1]}, "
                        f"segment start time: {t_start/1000}"
                    )

                audio_segment = audio[t_start:t_stop]
                out = audio_segment.export(str(tmp_path), format="wav")
                out.close()

        logging.shutdown()
        return dia_segments, None
    except Exception as e:
        error_text = f"segmenting the audio failed: {e}"
        return None, error_text
