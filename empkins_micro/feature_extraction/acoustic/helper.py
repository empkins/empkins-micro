import pandas as pd
import numpy as np
import more_itertools as mit


def process_segment_pitch(ff_df):
    # voice_label = ff_df['aco_voicelabel']
    #
    # indices_yes = [i for i, x in enumerate(voice_label) if x == "yes"]
    # voiced_yes = [list(group) for group in mit.consecutive_groups(indices_yes)]
    #
    # indices_no = [i for i, x in enumerate(voice_label) if x == "no"]
    # voiced_no = [list(group) for group in mit.consecutive_groups(indices_no)]
    #
    # com_speech = voiced_yes + voiced_no
    # com_speech_sort = sorted(com_speech, key=lambda x: x[0])

    # return com_speech_sort, voiced_yes, voiced_no


    voice_label = ff_df['aco_voicelabel']
    voice_label_yes = voice_label == "yes"
    voice_label_no = voice_label == "no"
    indices = np.arange(len(voice_label))
    indices_yes = indices[voice_label_yes]
    indices_no = indices[voice_label_no]
    diff_yes = np.diff(indices_yes)
    group_boundary_yes = np.where(diff_yes > 1)[0] + 1
    group_starts_yes = np.hstack([0, group_boundary_yes])
    group_ends_yes = np.hstack([group_boundary_yes, len(indices_yes)])
    voiced_yes = [list(indices_yes[start:end])
                  for start, end in zip(group_starts_yes, group_ends_yes)]
    diff_no = np.diff(indices_no)
    group_boundary_no = np.where(diff_no > 1)[0] + 1
    group_starts_no = np.hstack([0, group_boundary_no])
    group_ends_no = np.hstack([group_boundary_no, len(indices_no)])
    voiced_no = [list(indices_no[start:end])
                 for start, end in zip(group_starts_no, group_ends_no)]
    com_speech = voiced_yes + voiced_no
    com_speech_sort = np.array(sorted(com_speech, key=lambda x: x[0]))
    return com_speech_sort, voiced_yes, voiced_no