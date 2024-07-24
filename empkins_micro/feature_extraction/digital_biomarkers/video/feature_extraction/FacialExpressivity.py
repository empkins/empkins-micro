import pandas as pd
import torch
from feat import Detector


def calculate_pyfeat_features(
    videofile,
    face_model,
    landmark_model,
    au_model,
    emotion_model,
    facepose_model,
    skip_frames,
) -> pd.DataFrame:
    """
    Sample dataframe (FEX object) has the following features for each frame:
        - Frame number
        - Rectangle surrounding frame
        - X and Y values of facial landmarks
        - Action unit values
        - Emotion values
    """

    detector = Detector(
        face_model=face_model,
        landmark_model=landmark_model,
        au_model=au_model,
        emotion_model=emotion_model,
        facepose_model=facepose_model,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fex = detector.detect_video(
        str(videofile), device="cpu", skip_frames=skip_frames, pin_memory=True
    )  # , batch_size=8)

    return fex


def process_fer(
    videofile,
    face_model,
    landmark_model,
    au_model,
    emotion_model,
    facepose_model,
    skip_frames,
):
    pyfeat = calculate_pyfeat_features(
        videofile,
        face_model,
        landmark_model,
        au_model,
        emotion_model,
        facepose_model,
        skip_frames,
    )
    fer_emotions = pyfeat.emotions
    fer_aus = pyfeat.aus
    fer_faceBox = pyfeat.facebox
    euler_angles = pyfeat.facepose
    fer = pd.concat([fer_emotions, fer_aus, fer_faceBox], axis=1)
    return fer, euler_angles


# TODO: calculate emotional intensity
# TODO: add deepFace?
