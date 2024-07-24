import cv2
import mediapipe as mp
import pandas as pd

from empkins_micro.feature_extraction.digital_biomarkers.video.feature_extraction.helper import (
    calculate_range_of_motion,
    calculate_velocity,
    calculate_std_dev_movement,
    static_periods,
    below_threshold,
    visibility,
)


def calculate_mediapipe_features(videofile, skip_frames=0):
    """
    Calculate the mediapipe face mesh coordinates, hand landmarks, and pose landmarks for the entire video.

    Parameters:
    skip_frames (int): The number of frames to skip before processing the next frame.
    """

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    mediapipe_res = []
    hands_data = []
    pose_data = []
    # gaze_data = []

    cap = cv2.VideoCapture(str(videofile))

    with mp_hands.Hands() as hands, mp_pose.Pose() as pose, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process this frame if we have skipped enough frames
            if frame_count % (skip_frames + 1) == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_results = hands.process(image)
                pose_results = pose.process(image)
                face_results = face_mesh.process(image)

                mediapipe_res.append((face_results, frame_count))

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        hands_data.append((hand_landmarks, frame_count))

                if pose_results.pose_landmarks:
                    pose_data.append((pose_results.pose_landmarks, frame_count))

                # if face_results.multi_face_landmarks:
                #     for face_landmarks in face_results.multi_face_landmarks:
                #         gaze_data.append((face_landmarks, frame_count))

            frame_count += 1

    cap.release()

    pose_df = _get_pose_lmk_df(pose_data)
    hands_df = _get_hands_lmk_df(hands_data)
    mediapipe_df = _get_face_lmk_df(mediapipe_res)

    return mediapipe_df, hands_df, pose_df


def _get_face_lmk_df(mediapipe_res):
    # Convert output into dataframe
    landmark_cols = [["X" + str(i), "Y" + str(i), "Z" + str(i)] for i in range(468)]
    l_cols = []
    for elem in landmark_cols:
        l_cols.append(elem[0])
        l_cols.append(elem[1])
        l_cols.append(elem[2])
    l_cols.append("frame")
    mediapipe_df = pd.DataFrame(columns=l_cols)

    for frame, index in mediapipe_res:
        if frame.multi_face_landmarks:
            curr_data = [
                {"X" + str(i): l.x, "Y" + str(i): l.y, "Z" + str(i): l.z}
                for i, l in enumerate(frame.multi_face_landmarks[0].landmark)
            ]
            result = {}
            for d in curr_data:
                result.update(d)
            curr_df = pd.DataFrame([result])
            curr_df["frame"] = index
            mediapipe_df = pd.concat([mediapipe_df, curr_df], axis=0)
    return mediapipe_df


def _get_pose_lmk_df(pose_data):
    # LEFT_SHOULDER = 11
    # RIGHT_SHOULDER = 12
    # LEFT_ELBOW = 13
    # RIGHT_ELBOW = 14
    # 15 - 22 : left and right hands
    pose_df = pd.DataFrame()
    labels = [
        "leftShoulder",
        "rightShoulder",
        "leftElbow",
        "rightElbow",
        "leftWrist",
        "rightWrist",
        "leftPinky",
        "rightPinky",
        "leftIndex",
        "rightIndex",
        "leftThumb",
        "rightThumb",
    ]
    for pose_landmark in pose_data:
        landmarks = pose_landmark[0].landmark[11:23]
        data = {"frame": pose_landmark[1]}
        for label, lnd in zip(labels, landmarks):
            data.update(
                {
                    f"X_{label}": lnd.x,
                    f"Y_{label}": lnd.y,
                    f"Z_{label}": lnd.z,
                    f"visibility_{label}": lnd.visibility,
                }
            )
        df_tmp = pd.DataFrame([data])
        pose_df = pd.concat([pose_df, df_tmp])
    return pose_df


def _get_hands_lmk_df(hands_data):
    hands_df = pd.DataFrame()
    for hands_landmark in hands_data:
        landmarks = hands_landmark[0].landmark
        data = {"frame": hands_landmark[1]}
        for label, lnd in enumerate(landmarks):
            data.update(
                {
                    f"hands_X_{label}": lnd.x,
                    f"hands_Y_{label}": lnd.y,
                    f"hands_Z_{label}": lnd.z,
                    f"hands_visibility_{label}": lnd.visibility,
                }
            )
        df_tmp = pd.DataFrame([data])
        hands_df = pd.concat([hands_df, df_tmp])
    return hands_df


def get_movement(face, time_interval, pose=None, euler=None):
    if face.empty:
        return None

    regions = {
        "nose": range(1, 12),
        # 'forehead': range(127, 157),
        # 'cheekbones': range(159, 183)
    }

    features = {}
    for region, region_range in regions.items():
        # features[f'{region}_meanDistance'] = calculate_mean_distance(df, region_range)
        velocity = calculate_velocity(face, region_range)
        count = below_threshold(velocity)
        features[f"{region}_belowThresholdRatio"] = count / len(velocity)
        features[f"{region}_staticPeriods"] = static_periods(
            velocity, 1 / time_interval, threshold=0.05
        )
        features[f"{region}_meanAngularVelocity"] = velocity.mean()
        features[f"{region}_stdAngularVelocity"] = velocity.std()
        features[f"{region}_rangeOfMotion"] = calculate_range_of_motion(
            face, region_range
        )
        features[f"{region}_stdDevMovement"] = calculate_std_dev_movement(
            face, region_range
        )

    if pose is not None:
        regions = {
            "leftShoulder": ["_leftShoulder"],
            "rightShoulder": ["_rightShoulder"],
        }
        for region, region_range in regions.items():
            velocity = calculate_velocity(pose, region_range)
            features[f"{region}_belowThresholdRatio"] = below_threshold(velocity) / len(
                velocity
            )
            features[f"{region}_staticPeriods"] = static_periods(
                velocity, 1 / time_interval, threshold=0.05
            )
            features[f"{region}_rangeOfMotion"] = calculate_range_of_motion(
                pose, region_range
            )
            features[f"{region}_meanVelocity"] = velocity.mean()
            features[f"{region}_stdVelocity"] = velocity.std()
            features[f"{region}_stdDevMovement"] = calculate_std_dev_movement(
                pose, region_range
            )

        regions = {
            "leftElbow": ["_leftElbow"],
            "rightElbow": ["_rightElbow"],
            "leftHand": ["_leftWrist", "_leftPinky", "_leftIndex", "_leftThumb"],
            "rightHand": ["_rightWrist", "_rightPinky", "_rightIndex", "_rightThumb"],
        }
        for region, region_range in regions.items():
            features[f"{region}_visibility"] = visibility(pose, region_range)

    # Add mean euler angles
    mean_euler = process_euler_angles(euler=euler)
    features.update(mean_euler)

    return pd.DataFrame(features, index=[0])


def process_euler_angles(euler):
    euler_mean = euler.mean()
    return euler_mean.to_dict()
