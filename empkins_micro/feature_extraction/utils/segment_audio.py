import pandas as pd
from pathlib import Path
import numpy as np
import os
import shutil
from pydub import AudioSegment

def clean_diarization(diarization):
    dia = diarization.iloc[np.where(diarization["speaker"] == "SPEAKER_PANEL_INV")[0]]
    dia = dia.reset_index(drop=True)
    return dia

def segment_audio(audio_path: Path, subject_id: str, condition: str, diarization: pd.DataFrame, files_path: Path):
    dia_segments = clean_diarization(diarization=diarization)
    if files_path.exists():
        shutil.rmtree(files_path)
        os.mkdir(files_path)
    else:
        os.mkdir(files_path)

    with open(audio_path, "rb") as audio_file:
        audio = AudioSegment.from_file(audio_file, format="wav")

        for idx, seg in dia_segments.iterrows():
            tmp_path = files_path.joinpath(f"{subject_id}_{condition}_seg_{idx}.wav")
            audio_segment = audio[seg['start']:seg['stop']]
            out = audio_segment.export(tmp_path, format="wav")
            out.close()



    return dia_segments