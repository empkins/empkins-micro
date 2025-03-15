import numpy as np
import pandas as pd

from empkins_micro.feature_extraction.digital_biomarkers.acoustic.helper import get_length


def _audio_shimmer(sound):
    """
    Using parselmouth library fetching shimmer
    Args:
        sound: parselmouth object
    Returns:
        (list) list of shimmers for each voice frame
    """
    try:
        import parselmouth
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Module 'parselmouth' not found. Please install it manually or "
                                  "install 'empkins-micro' with 'audio' extras via "
                                  "'poetry install empkins_micro -E audio'") from e
    pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)...", 80, 500)
    shimmer = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return shimmer


def _segment_shimmer(com_speech_sort, shimmer_frames, audio_file):
    """
    calculating shimmer for each voice segment
    """

    snd = parselmouth.Sound(str(audio_file))
    pitch = snd.to_pitch(time_step=0.001)

    for idx, vs in enumerate(com_speech_sort):
        shimmer = np.NaN
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
                shimmer = _audio_shimmer(samples)
            else:
                error = "voice segment is to short for shimmer calculation"

        except Exception as e:
            error = f"shimmer calculation failed for this segment: {e}"

        shimmer_frames[idx] = [shimmer, start_time, end_time, error]
    return shimmer_frames


def empty_shimmer(error_text):

    data = {"aco_shimmer": [np.nan], "start_time": [np.nan], "end_time": [np.nan], "error": [error_text]}
    return pd.DataFrame.from_dict(data)


def calc_shimmer(audio_file, voice_seg):
    """
    Preparing shimmer matrix
    Args:
        audio_file: (.wav) parsed audio file
    """

    try:

        audio_duration = get_length(audio_file)

        if float(audio_duration) < 0.064:
            return empty_shimmer("audio duration less than 0.064 seconds")

        cols_out = ["aco_shimmer", "start_time", "end_time", "error"]

        shimmer_frames = [[np.NaN for _ in cols_out]] * len(voice_seg)
        shimmer_segment_frames = _segment_shimmer(voice_seg, shimmer_frames, audio_file)

        df_shimmer = pd.DataFrame(shimmer_segment_frames, columns=cols_out)

        return df_shimmer

    except Exception as e:
        return empty_shimmer(f"failed to process audio file: {e}")
