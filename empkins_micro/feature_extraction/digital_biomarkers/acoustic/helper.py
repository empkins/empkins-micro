import subprocess

import numpy as np


def process_segment_pitch(ff_df):
    try:
        voice_label = ff_df["aco_voicelabel"]
        voice_label_yes = voice_label == "yes"

        indices = np.arange(len(voice_label))
        indices_yes = indices[voice_label_yes]

        diff_yes = np.diff(indices_yes)
        group_starts_yes = np.hstack([0, np.where(diff_yes > 1)[0] + 1])
        group_ends_yes = np.hstack([np.where(diff_yes > 1)[0] + 1, len(indices_yes)])
        voiced_yes = [list(indices_yes[start:end]) for start, end in zip(group_starts_yes, group_ends_yes)]

        return voiced_yes, None
    except Exception as e:
        error_text = f"pitch dataframe is malfunctioning: {e}"
        return None, error_text


def get_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    return float(result.stdout)
