import numpy as np
import pandas as pd
import smallestenclosingcircle

from empkins_micro.feature_extraction.digital_biomarkers.video.feature_extraction.helper import (
    calculate_velocity,
    static_periods,
    below_threshold,
)


def calculate_pupil_features(mediapipe_features, time=1):
    """
    Calculate the radius of pupil size for each eye based on this tutorial on Mediapipe landmarks:
    https://medium.com/mlearning-ai/iris-segmentation-mediapipe-python-a4deb711aae3
    """
    if mediapipe_features.empty:
        return None

    left_iris = ["474", "475", "476", "477"]
    right_iris = ["469", "470", "471", "472"]

    left_eye_upper = ["386", "374"]
    left_eye_lower = ["387", "380"]
    left_eye_horizontal = ["263", "362"]

    right_eye_upper = ["159", "145"]
    right_eye_lower = ["160", "153"]
    right_eye_horizontal = ["133", "33"]

    left_eye = left_eye_upper + left_eye_lower + left_eye_horizontal
    right_eye = right_eye_upper + right_eye_lower + right_eye_horizontal

    # Define the EAR threshold for blink detection
    EAR_THRESHOLD = 0.47  # This is an example value; it might need adjustment

    # Initialize counters and lists
    blink_count_left = 0
    blink_count_right = 0
    ear_left_list = []
    ear_right_list = []

    radi_l = []
    radi_r = []
    for i in range(len(mediapipe_features)):
        l_xdat = mediapipe_features[["X" + elem for elem in left_iris]].iloc[i]
        l_ydat = mediapipe_features[["Y" + elem for elem in left_iris]].iloc[i]
        _, _, l_rad = smallestenclosingcircle.make_circle(list(zip(l_xdat, l_ydat)))

        r_xdat = mediapipe_features[["X" + elem for elem in right_iris]].iloc[i]
        r_ydat = mediapipe_features[["Y" + elem for elem in right_iris]].iloc[i]
        _, _, r_rad = smallestenclosingcircle.make_circle(list(zip(r_xdat, r_ydat)))

        radi_l.append(l_rad)
        radi_r.append(r_rad)

        # Extract eye landmarks for EAR calculation
        left_eye_points = np.array(
            [
                mediapipe_features[["X" + num, "Y" + num]].iloc[i].to_numpy()
                for num in left_eye
            ]
        )
        right_eye_points = np.array(
            [
                mediapipe_features[["X" + num, "Y" + num]].iloc[i].to_numpy()
                for num in right_eye
            ]
        )

        # Calculate EAR for each eye
        ear_left = calculate_EAR(left_eye_points)
        ear_right = calculate_EAR(right_eye_points)

        ear_left_list.append(ear_left)
        ear_right_list.append(ear_right)

        # Detect blinks for each eye
        if ear_left < EAR_THRESHOLD:
            blink_count_left += 1
        if ear_right < EAR_THRESHOLD:
            blink_count_right += 1

    results = {}

    left_vel = calculate_velocity(
        mediapipe_features, left_iris, time, correction=left_eye
    )
    right_vel = calculate_velocity(
        mediapipe_features, right_iris, time, correction=right_eye
    )

    results["left_staticPeriods2"] = static_periods(left_vel, 1 / time, threshold=0.2)
    results["left_staticPeriods1"] = static_periods(left_vel, 1 / time, threshold=0.1)
    results["left_staticPeriods05"] = static_periods(left_vel, 1 / time, threshold=0.05)
    results["left_staticPeriods01"] = static_periods(left_vel, 1 / time, threshold=0.01)
    results["left_belowThres"] = below_threshold(left_vel) / len(left_vel)

    results["right_staticPeriods2"] = static_periods(right_vel, 1 / time, threshold=0.2)
    results["right_staticPeriods1"] = static_periods(right_vel, 1 / time, threshold=0.1)
    results["right_staticPeriods05"] = static_periods(
        right_vel, 1 / time, threshold=0.05
    )
    results["right_staticPeriods01"] = static_periods(
        right_vel, 1 / time, threshold=0.01
    )
    results["right_belowThres"] = below_threshold(right_vel) / len(right_vel)

    results["left_velocity_mean"] = left_vel.mean()
    results["left_velocity_std"] = left_vel.std()
    results["right_velocity_mean"] = right_vel.mean()
    results["right_velocity_std"] = right_vel.std()

    pupils = pd.DataFrame({"left_pupil": radi_l, "right_pupil": radi_r})
    results["left_radi_mean"] = pupils["left_pupil"].mean()
    results["right_radi_mean"] = pupils["right_pupil"].mean()
    results["left_radi_std"] = pupils["left_pupil"].std()
    results["right_radi_std"] = pupils["right_pupil"].std()
    results["left_radi_diff"] = pupils["left_pupil"].diff().fillna(0).mean()
    results["right_radi_diff"] = pupils["right_pupil"].diff().fillna(0).mean()
    results["left_radi_max"] = pupils["left_pupil"].max()
    results["right_radi_max"] = pupils["right_pupil"].max()
    results["left_radi_min"] = pupils["left_pupil"].min()
    results["right_radi_min"] = pupils["right_pupil"].min()

    results["left_blinks"] = blink_count_left
    results["right_blinks"] = blink_count_right

    return pd.DataFrame(results, index=[0])


def calculate_EAR(eye_points):
    # Compute the distances between vertical landmarks and horizontal landmarks
    vert_dist = np.linalg.norm(eye_points[0] - eye_points[1]) + np.linalg.norm(
        eye_points[2] - eye_points[3]
    )
    horiz_dist = np.linalg.norm(eye_points[4] - eye_points[5])

    # Calculate EAR
    ear = vert_dist / (2.0 * horiz_dist)
    return ear
