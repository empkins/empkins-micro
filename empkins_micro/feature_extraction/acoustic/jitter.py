import pandas as pd
import numpy as np

import parselmouth

def _audio_jitter(sound):
    """
    Using parselmouth library fetching jitter
    Args:
        sound: parselmouth object
    Returns:
        (list) list of jitters for each voice frame
    """
    pointProcess = parselmouth.praat.call(
        sound, "To PointProcess (periodic, cc)...", 80, 500
    )
    jitter = parselmouth.praat.call(
        pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    return jitter

def _segment_jitter(com_speech_sort, jitter_frames, audio_file):
    """
    calculating jitter for each voice segment
    """
    snd = parselmouth.Sound(str(audio_file))
    pitch = snd.to_pitch(time_step=0.001)

    for idx, vs in enumerate(com_speech_sort):
        try:
            jitter = np.NaN
            start_time = np.NaN
            end_time = np.NaN

            if len(vs) > 1:

                start_time = pitch.get_time_from_frame_number(vs[0])
                end_time = pitch.get_time_from_frame_number(vs[-1])

                snd_start = int(snd.get_frame_number_from_time(start_time))
                snd_end = int(snd.get_frame_number_from_time(end_time))

                samples = parselmouth.Sound(snd.as_array()[0][snd_start:snd_end])
                jitter = _audio_jitter(samples)
        except:
            pass

        jitter_frames[idx] = [jitter, start_time, end_time]
    return jitter_frames


def calc_jitter(audio_file, voice_seg):
    """
    Preparing jitter matrix
    Args:
        audio_file: (.wav) parsed audio file
    """

    cols_out = ['aco_jitter', 'start_time', 'end_time']

    jitter_frames = [[np.NaN for _ in cols_out]] * len(voice_seg)

    jitter_segment_frames = _segment_jitter(
        voice_seg, jitter_frames, audio_file
    )

    df_jitter = pd.DataFrame(jitter_segment_frames, columns=cols_out)

    return df_jitter