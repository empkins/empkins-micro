import numpy as np
import pandas as pd
import parselmouth

from empkins_micro.feature_extraction.digital_biomarkers.acoustic.helper import get_length


def audio_pitch_frame(pitch):
    """
    Computing total number of speech and participant voiced frames
    Args:
        pitch: speech pitch
    Returns:
        (float) total voice frames and participant voiced frames
    """
    total_frames = pitch.get_number_of_frames()
    voiced_frames = pitch.count_voiced_frames()
    return total_frames, voiced_frames


def voice_segment(path):
    """
    Using parselmouth library for fundamental frequency
    Args:
        path: (.wav) audio file location
    Returns:
        (float) total voice frames, participant voiced frames and voiced frames percentage
    """
    sound_pat = parselmouth.Sound(str(path))
    pitch = sound_pat.to_pitch()
    total_frames, voiced_frames = audio_pitch_frame(pitch)

    voiced_percentage = (voiced_frames / total_frames) * 100
    res_dict = {"aco_voicepct": [voiced_percentage], "error": ["PASS"]}
    return res_dict


def empty_vfs(error_text):
    data = {"aco_voicepct": [np.nan], "error": [error_text]}
    return pd.DataFrame.from_dict(data)


def calc_vfs(audio_file):
    """
    creating dataframe matrix for voice frame score
    Args:
        audio_file: Audio file path
        new_out_base_dir: AWS instance output base directory path
        f_nm_config: Config file object
    """
    try:
        audio_duration = get_length(audio_file)

        if float(audio_duration) < 0.064:
            return empty_vfs("audio duration less than 0.064 seconds")

        voice_percentage_dict = voice_segment(audio_file)

        df = pd.DataFrame.from_dict(voice_percentage_dict)
        return df
    except Exception as e:
        return empty_vfs(f"failed to process audio file: {e}")
