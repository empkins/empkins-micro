import parselmouth
from parselmouth.praat import run_file
import re
import json
import pandas as pd
import os
from pathlib import Path

# Executing praat script using parselmouth function
def calc_voicetremor(snd_file):
    """
    Generating Voice tremor endpoint dataframe
    Args:
        snd_file: (.wav) parsed audio file
    Returns tremor endpoint dataframe
    """

    dmlib_path = Path(os.path.realpath(os.path.dirname(__file__)))
    dmlib_path = dmlib_path.parent
    dmlib_path = str(dmlib_path.joinpath("resources", "voice_tremor.praat"))

    snd = parselmouth.Sound(str(snd_file))
    try:
        tremor_var = run_file(snd, dmlib_path, capture_output=True)
    except:
        print("audio duration is shorter than 0.5 seconds")
        return ""
    new_tremor_var = re.sub("--undefined--", "0", tremor_var[1])
    res = json.loads(new_tremor_var)

    tremor_df = pd.DataFrame(res, index=["0"])
    tremor_df.columns = ["mov_freqtremfreq", "mov_amptremfreq", "mov_freqtremindex",
                         "mov_amptremindex", "mov_freqtrempindex", "mov_amptrempindex"]

    return tremor_df
