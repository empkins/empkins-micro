import pandas as pd
import numpy as np

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

def _segment_shimmer(com_speech_sort, shimmer_frames, audio_file):
    """
    calculating shimmer for each voice segment
    """

    snd = parselmouth.Sound(str(audio_file))
    pitch = snd.to_pitch(time_step=0.001)

    for idx, vs in enumerate(com_speech_sort):
        try:
            shimmer = np.NaN
            start_time = np.NaN
            end_time = np.NaN

            if len(vs) > 1:

                start_time = pitch.get_time_from_frame_number(vs[0])
                end_time = pitch.get_time_from_frame_number(vs[-1])

                snd_start = int(snd.get_frame_number_from_time(start_time))
                snd_end = int(snd.get_frame_number_from_time(end_time))

                samples = parselmouth.Sound(snd.as_array()[0][snd_start:snd_end])
                shimmer = _audio_shimmer(samples)
        except:
            pass

        shimmer_frames[idx] = [shimmer, start_time, end_time]
    return shimmer_frames


def calc_shimmer(audio_file, voice_seg):
    """
    Preparing shimmer matrix
    Args:
        audio_file: (.wav) parsed audio file
    """

    cols_out = ['aco_shimmer', 'start_time', 'end_time']

    shimmer_frames = [[np.NaN for _ in cols_out]] * len(voice_seg)
    shimmer_segment_frames = _segment_shimmer(
        voice_seg, shimmer_frames, audio_file
    )

    df_shimmer = pd.DataFrame(
        shimmer_segment_frames, columns=cols_out
    )

    return df_shimmer
