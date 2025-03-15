import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from empkins_micro.feature_extraction.digital_biomarkers.acoustic.helper import get_length


def empty_voicetremor(error_text):
    data = {
        "mov_freqtremfreq": [np.nan],
        "mov_amptremfreq": [np.nan],
        "mov_freqtremindex": [np.nan],
        "mov_amptremindex": [np.nan],
        "mov_freqtrempindex": [np.nan],
        "mov_amptrempindex": [np.nan],
        "error": [error_text],
    }
    return pd.DataFrame.from_dict(data)


def calc_voicetremor(snd_file):
    """
    Generating Voice tremor endpoint dataframe
    Args:
        snd_file: (.wav) parsed audio file
    Returns tremor endpoint dataframe
    """
    try:
        import parselmouth
        from parselmouth.praat import run_file
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Module 'parselmouth' not found. Please install it manually or "
                                  "install 'empkins-micro' with 'audio' extras via "
                                  "'poetry install empkins_micro -E audio'") from e
    try:

        audio_duration = get_length(snd_file)

        if float(audio_duration) < 0.5:
            return empty_voicetremor("audio duration less than 0.5 seconds")

        dmlib_path = Path(os.path.realpath(os.path.dirname(__file__)))
        dmlib_path = dmlib_path.parent
        dmlib_path = str(dmlib_path.joinpath("resources", "voice_tremor.praat"))

        snd = parselmouth.Sound(str(snd_file))
        tremor_var = run_file(snd, dmlib_path, capture_output=True)

        new_tremor_var = re.sub("--undefined--", "0", tremor_var[1])
        res = json.loads(new_tremor_var)

        tremor_dict = {
            "mov_freqtremfreq": [res["FTrF"]],
            "mov_amptremfreq": [res["ATrF"]],
            "mov_freqtremindex": [res["FTrI"]],
            "mov_amptremindex": [res["ATrI"]],
            "mov_freqtrempindex": [res["FTrP"]],
            "mov_amptrempindex": [res["ATrP"]],
            "error": ["PASS"],
        }

        return pd.DataFrame.from_dict(tremor_dict)

    except Exception as e:
        return empty_voicetremor(f"failed to process audio file: {e}")
