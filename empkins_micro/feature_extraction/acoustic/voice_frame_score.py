import pandas as pd
import parselmouth


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
    try:
        pitch = sound_pat.to_pitch()
    except:
        print("audio duration is shorter than 0.064 seconds")
        return ""

    total_frames, voiced_frames = audio_pitch_frame(pitch)

    voiced_percentage = (voiced_frames / total_frames) * 100
    return voiced_percentage

def calc_vfs(audio_file):
    """
    creating dataframe matrix for voice frame score
    Args:
        audio_file: Audio file path
        new_out_base_dir: AWS instance output base directory path
        f_nm_config: Config file object
    """

    voice_percentage = voice_segment(audio_file)
    if isinstance(voice_percentage, str):
        return ""

    df = pd.DataFrame(data=[voice_percentage], columns=['aco_voicepct'])
    return df