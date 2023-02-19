import numpy as np
import json
import re
from pathlib import Path
import pandas as pd
import os

def euclidean_distance(point1, point2):
    """
    Compute euclidean distance between points
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def expand_landmarks(landmarks):
    """
    util method to expand landmark list:
    eg: [1,2] -> [['l1_x', 'l1_y'], ['l2_x', 'l2_y']]
    """
    return [["l{}_x".format(point), "l{}_y".format(point)] for point in landmarks]


def calc_displacement_vec(df, landmarks, num_frames):
    """
    Calculates displacement vector frame by frame
    """

    landmarks = expand_landmarks(landmarks)

    disp_vec = np.zeros((len(landmarks), num_frames))
    prev_point = np.zeros((len(landmarks), 2))

    # initialize
    for j, pair in enumerate(landmarks):
        first_row = df.iloc[0]
        prev_point[j] = (first_row[pair[0]], first_row[pair[1]])

    for i in range(num_frames):
        frame_row = df.iloc[i]
        for j, pair in enumerate(landmarks):
            x, y = pair[0], pair[1]
            current = (frame_row[x], frame_row[y])
            deviation = euclidean_distance(current, prev_point[j])
            disp_vec[j][i] = deviation
            prev_point[j] = current

    return disp_vec


def compute_features(df_of, config_path):
    """Computes features
    Returns: features in vector format
    """

    config = json.loads(open(config_path, "r").read())

    pattern_x = re.compile(r"l\d+_x")
    pattern_y = re.compile(r"l\d+_y")

    # assumption: distance of face to camera remains at roughly static

    # logic break
    landmark_columns = []
    for col in df_of.columns:
        if pattern_x.match(col) or pattern_y.match(col):
            landmark_columns.append(col)

    df_of = df_of[(df_of[landmark_columns] != 0).any(axis=1)]
    df_of.reset_index(inplace=True)

    num_frames = len(df_of)
    landmarks = config["landmarks"]

    try:

        if num_frames == 0:
            return empty_facial_tremor("no frames with visible face")

        first_row = df_of.iloc[0]

        facew = abs(
            first_row[config["face_width_left"]] - first_row[config["face_width_right"]]
        )
        faceh = abs(
            first_row[config["face_height_left"]]
            - first_row[config["face_height_right"]]
        )

        if facew == 0 or faceh == 0:
            return empty_facial_tremor("face width or height = 0. Check landmark values")

        fac_disp = calc_displacement_vec(df_of, landmarks, num_frames)

        if len(fac_disp.shape) != 2:
            return empty_facial_tremor("fac_disp is not 2D. smth went wrong with disp calc")

        if len(fac_disp[0]) <= 1:
            return empty_facial_tremor("Video too short. smth went wrong with disp calc")

        fac_disp_median = np.median(fac_disp, axis=1)

        fac_corr_mat = np.corrcoef(fac_disp, rowvar=True)

        # extract relevant row from cov matrix
        ref_lmk_index = [
            i for i, lmk in enumerate(landmarks) if config["ref_lmk"] == lmk
        ]

        fac_corr = fac_corr_mat[ref_lmk_index][0]
        fac_area = config["ref_area"] / (facew * faceh)

        fac_features_median = np.multiply(fac_area * fac_disp_median, (1.0 - fac_corr))

        fac_features_dict = {}
        for i, landmark in enumerate(landmarks):

            fac_features_dict["fac_disp_median_{}_mean".format(landmark)] = [
                fac_features_median[i]
            ]

        fac_features_dict["error"] = ["PASS"]
        data = pd.DataFrame.from_dict(fac_features_dict)
        return data

    except Exception as e:
        return empty_facial_tremor(f"error while computing facial tremor features: {e}")


def empty_facial_tremor(error_text):
    data = {
        "fac_tremor_median_5_mean": [np.nan],
        "fac_tremor_median_12_mean": [np.nan],
        "fac_tremor_median_8_mean": [np.nan],
        "fac_tremor_median_48_mean": [np.nan],
        "fac_tremor_median_54_mean": [np.nan],
        "fac_tremor_median_28_mean": [np.nan],
        "fac_tremor_median_51_mean": [np.nan],
        "fac_tremor_median_66_mean": [np.nan],
        "fac_tremor_median_57_mean": [np.nan],
        "error": [error_text]
    }
    return pd.DataFrame.from_dict(data)


def calc_facial_tremor(df_of):
    try:
        config_path = Path(os.path.realpath(os.path.dirname(__file__)))
        config_path = config_path.parent
        config_path = str(config_path.joinpath("resources", "facial_tremor_config.json"))
        return compute_features(df_of, config_path)
    except Exception as e:
        return empty_facial_tremor(f"failed to process video file for facial_tremor: {e}")
