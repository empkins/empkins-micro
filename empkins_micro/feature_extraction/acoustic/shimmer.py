import pandas as pd
import numpy as np
from empkins_micro.feature_extraction.acoustic.helper import process_segment_pitch
import parselmouth

def _audio_shimmer(sound):
    """
    Using parselmouth library fetching shimmer
    Args:
        sound: parselmouth object
    Returns:
        (list) list of shimmers for each voice frame
    """
    pointProcess = parselmouth.praat.call(
        sound, "To PointProcess (periodic, cc)...", 80, 500
    )
    shimmer = parselmouth.praat.call(
        [sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    return shimmer

def _segment_shimmer(com_speech_sort, voiced_yes, voiced_no, shimmer_frames, audio_file):
    """
    calculating shimmer for each voice segment
    """

    snd = parselmouth.Sound(str(audio_file))
    pitch = snd.to_pitch(time_step=0.001)

    for idx, vs in enumerate(com_speech_sort):
        try:

            shimmer = np.NaN
            if vs in voiced_yes and len(vs) > 1:

                start_time = pitch.get_time_from_frame_number(vs[0])
                end_time = pitch.get_time_from_frame_number(vs[-1])

                snd_start = int(snd.get_frame_number_from_time(start_time))
                snd_end = int(snd.get_frame_number_from_time(end_time))

                samples = parselmouth.Sound(snd.as_array()[0][snd_start:snd_end])
                shimmer = _audio_shimmer(samples)
        except:
            pass

        shimmer_frames[idx] = shimmer
    return shimmer_frames


def calc_shimmer(ff_df, audio_file):
    """
    Preparing shimmer matrix
    Args:
        audio_file: (.wav) parsed audio file

    """

    voice_seg = process_segment_pitch(ff_df)

    shimmer_frames = [np.NaN] * len(voice_seg[0])
    shimmer_segment_frames = _segment_shimmer(
        voice_seg[0], voice_seg[1], voice_seg[2], shimmer_frames, audio_file
    )

    df_shimmer = pd.DataFrame(
        shimmer_segment_frames, columns=['aco_shimmer']
    )

    return df_shimmer
