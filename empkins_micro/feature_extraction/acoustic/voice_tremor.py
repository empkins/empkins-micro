import parselmouth
from parselmouth.praat import run_file
import re
import json
import pandas as pd

# Executing praat script using parselmouth function
def tremor_praat(snd_file):
    """
    Generating Voice tremor endpoint dataframe
    Args:
        snd_file: (.wav) parsed audio file
        r_cfg: Raw variable configuration file
    Returns tremor endpoint dataframe
    """

    dmlib_path = "C:/Users/marie/repositories/open_dbm/opendbm/resources/libraries/voice_tremor.praat"

    snd = parselmouth.Sound(snd_file)
    tremor_var = run_file(snd, dmlib_path, capture_output=True)
    new_tremor_var = re.sub("--undefined--", "0", tremor_var[1])
    res = json.loads(new_tremor_var)

    tremor_df = pd.DataFrame(
        res,
        index=[
            "0",
        ],
    )

    tremor_df.columns = [
        "mov_freqtremfreq_mean", "mov_amptremfreq_mean", "mov_freqtremindex_mean", "mov_amptremindex_mean", "mov_freqtrempindex_mean",
        "mov_amptrempindex_mean" ]


    return tremor_df
