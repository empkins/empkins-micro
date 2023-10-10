import numpy as np
import pandas as pd

from empkins_micro.feature_extraction.digital_biomarkers.acoustic.helper import get_length


def _audio_jitter(sound):
    """
    Using parselmouth library fetching jitter
    Args:
        sound: parselmouth object
    Returns:
        (list) list of jitters for each voice frame
    """
    try:
        import parselmouth
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Module 'parselmouth' not found. Please install it manually or "
                                  "install 'empkins-micro' with 'audio' extras via "
                                  "'poetry install empkins_micro -E audio'") from e
    pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)...", 80, 500)
    jitter = parselmouth.praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return jitter


def _segment_jitter(com_speech_sort, jitter_frames, audio_file):
    """
    calculating jitter for each voice segment
    """
    snd = parselmouth.Sound(str(audio_file))
    pitch = snd.to_pitch(time_step=0.001)

    for idx, vs in enumerate(com_speech_sort):
        jitter = np.NaN
        start_time = np.NaN
        end_time = np.NaN
        error = "PASS"
        try:
            if len(vs) > 1:
                start_time = pitch.get_time_from_frame_number(vs[0])
                end_time = pitch.get_time_from_frame_number(vs[-1])

                snd_start = int(snd.get_frame_number_from_time(start_time))
                snd_end = int(snd.get_frame_number_from_time(end_time))

                samples = parselmouth.Sound(snd.as_array()[0][snd_start:snd_end])
                jitter = _audio_jitter(samples)
            else:
                error = "voice segment is to short for jitter calculation"
        except Exception as e:
            error = f"jitter calculation failed for this segment: {e}"

        jitter_frames[idx] = [jitter, start_time, end_time, error]
    return jitter_frames


def empty_jitter(error_text):
    data = {"aco_jitter": [np.nan], "start_time": [np.nan], "end_time": [np.nan], "error": [error_text]}
    return pd.DataFrame.from_dict(data)


def calc_jitter(audio_file, voice_seg):
    """
    Preparing jitter matrix
    Args:
        audio_file: (.wav) parsed audio file
    """

    try:
        audio_duration = get_length(audio_file)
        if float(audio_duration) < 0.064:
            return empty_jitter("audio duration less than 0.064 seconds")

        cols_out = ["aco_jitter", "start_time", "end_time", "error"]

        jitter_frames = [[np.NaN for _ in cols_out]] * len(voice_seg)

        jitter_segment_frames = _segment_jitter(voice_seg, jitter_frames, audio_file)

        df_jitter = pd.DataFrame(jitter_segment_frames, columns=cols_out)
        return df_jitter
    except Exception as e:
        return empty_jitter(f"failed to process audio file: {e}")
