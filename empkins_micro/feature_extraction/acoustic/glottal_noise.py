import pandas as pd
import numpy as np

import parselmouth

def _gne_ratio(sound):
    """
    Using parselmouth library fetching glottal noise excitation ratio
    Args:
        sound: parselmouth object
    Returns:
        (list) list of gne ratio for each voice frame
    """
    harmonicity_gne = sound.to_harmonicity_gne()
    gne_all_bands = harmonicity_gne.values
    gne_all_bands = np.where(gne_all_bands == -200, np.NaN, gne_all_bands)

    gne = np.nanmax(
        gne_all_bands
    )  # following http://www.fon.hum.uva.nl/rob/NKI_TEVA/TEVA/HTML/NKI_TEVA.pdf
    return gne


def _segment_gne(com_speech_sort, gne_all_frames, audio_file):
    """
    calculating gne for each voice segment
    """
    snd = parselmouth.Sound(str(audio_file))
    pitch = snd.to_pitch(time_step=0.001)

    for idx, vs in enumerate(com_speech_sort):
        try:
            max_gne = np.NaN
            start_time = np.NaN
            end_time = np.NaN

            if len(vs) > 1:

                start_time = pitch.get_time_from_frame_number(vs[0])
                end_time = pitch.get_time_from_frame_number(vs[-1])

                snd_start = int(snd.get_frame_number_from_time(start_time))
                snd_end = int(snd.get_frame_number_from_time(end_time))

                samples = parselmouth.Sound(snd.as_array()[0][snd_start:snd_end])
                max_gne = _gne_ratio(samples)
        except:
            pass

        gne_all_frames[idx] = [max_gne, start_time, end_time]
    return gne_all_frames


def calc_gne(audio_file, voice_seg):
    """
    Preparing gne matrix
    Args:
        audio_file: (.wav) parsed audio file
    """

    cols_out = ['aco_gne', 'start_time', 'end_time']

    gne_all_frames = [[np.NaN for _ in cols_out]] * len(voice_seg)
    gne_segment_frames = _segment_gne(
        voice_seg, gne_all_frames, audio_file
    )

    df_gne = pd.DataFrame(gne_segment_frames, columns=cols_out)

    return df_gne
