import numpy as np
import pandas as pd

from empkins_micro.feature_extraction.digital_biomarkers.video.feature_extraction import (
    FacialExpressivity as fe,
)
from empkins_micro.feature_extraction.digital_biomarkers.video.feature_extraction import (
    GazeBehavior as gaze,
)
from empkins_micro.feature_extraction.digital_biomarkers.video.feature_extraction import (
    Movement as mov,
)


class FacialFeatures:
    """
    There are a large list of features for both libraries used for this portion of the project.
    """

    videofile: str
    # pyfeat_features: data.Fex
    mediapipe_features: pd.DataFrame  # List of Mediapipe landmarks and what they correspond to: https://bit.ly/3wRUxAG
    pupil_radii: np.array
    summary_df: pd.DataFrame

    def __init__(
        self,
        videofile,
        output_path,
        save_output=True,
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        skip_frames=30,
        sample_rate=30,
    ):

        self._videofile = videofile
        self._save_output = save_output
        self._output_path = output_path
        self._face_model = face_model
        self._landmark_model = landmark_model
        self._au_model = au_model
        self._emotion_model = emotion_model
        self._facepose_model = facepose_model
        self._skip_frames = skip_frames
        self._sample_rate = sample_rate

    def save_results(self):
        output_name = self._videofile.stem
        if self._fer_features is not None:
            self._fer_features.to_csv(
                self._output_path.joinpath(f"pyfeat_{output_name}.csv")
            )

        if self._mediapipe_features is not None:
            self._mediapipe_features.to_csv(
                self._output_path.joinpath(f"mediapipe_{output_name}.csv")
            )

        if self._hands_df is not None:
            self._hands_df.to_csv(
                self._output_path.joinpath(f"mp_hands_{output_name}.csv")
            )

        if self._pose_df is not None:
            self._pose_df.to_csv(
                self._output_path.joinpath(f"mp_pose_{output_name}.csv")
            )

        if self._movement_features is not None:
            self._movement_features.to_csv(
                self._output_path.joinpath(f"movement_{output_name}.csv")
            )

        if self._pupil_radii is not None:
            self._pupil_radii.to_csv(
                self._output_path.joinpath(f"pupilRadii_{output_name}.csv")
            )

        self._summary_dataframe.to_csv(
            self._output_path.joinpath(f"summary_{output_name}.csv")
        )
        print(f"Results saved to {self._output_path}")

    def process(self):
        self._fer_features, self._euler_angles = fe.process_fer(
            self._videofile,
            face_model=self._face_model,
            landmark_model=self._landmark_model,
            au_model=self._au_model,
            emotion_model=self._emotion_model,
            facepose_model=self._facepose_model,
            skip_frames=self._skip_frames,
        )
        print("Finished PyFeat")
        (
            self._mediapipe_features,
            self._hands_df,
            self._pose_df,
        ) = mov.calculate_mediapipe_features(
            self._videofile, skip_frames=self._skip_frames // 3
        )
        time_interval = (self._skip_frames // 3) / self._sample_rate
        self._movement_features = mov.get_movement(
            face=self._mediapipe_features,
            time_interval=time_interval,
            pose=self._pose_df,
            euler=self._euler_angles,
        )

        print("Finished Mediapipe")
        self._pupil_radii = gaze.calculate_pupil_features(
            self._mediapipe_features, time=time_interval
        )
        print("Finished pupil")
        self._construct_summary_dataframe()
        if self._save_output:
            self.save_results()

    @property
    def summary_dataframe(self):
        return self._summary_dataframe

    @property
    def pyfeat_features(self):
        return self._pyfeat_features

    @property
    def mediapipe_features(self):
        return self._mediapipe_features

    @property
    def pupil_radii(self):
        return self._pupil_radii

    def _construct_summary_dataframe(self):
        """
        Calculate mean and standard deviation of all columns and return them in a DataFrame.
        """
        pyfeat_summary = self._calculate_metrics(self._fer_features)
        mediapipe_summary = self._calculate_metrics(self._mediapipe_features)
        mp_hands_summary = self._calculate_metrics(self._hands_df)
        mp_pose_summary = self._calculate_metrics(self._pose_df)

        self._summary_dataframe = pd.concat(
            [
                pyfeat_summary,
                mediapipe_summary,
                mp_hands_summary,
                mp_pose_summary,
                self._movement_features,
                self._pupil_radii,
            ],
            axis=1,
        )  # ,

    @staticmethod
    def _calculate_metrics(df):
        col_list = [
            col for col in df.columns if col not in ["input", "approx_time", "frame"]
        ]
        means = [(col + "_mean", df[col].mean()) for col in col_list]
        stds = [(col + "_std", df[col].std()) for col in col_list]
        max = [(col + "_max", df[col].max()) for col in col_list]  # _MAX
        min = [(col + "_min", df[col].min()) for col in col_list]  # _MIN
        mean_df = pd.DataFrame(
            [[elem[1] for elem in means]],
            index=[0],
            columns=[elem[0] for elem in means],
        )
        std_df = pd.DataFrame(
            [[elem[1] for elem in stds]], index=[0], columns=[elem[0] for elem in stds]
        )
        max_df = pd.DataFrame(
            [[elem[1] for elem in max]], index=[0], columns=[elem[0] for elem in max]
        )
        min_df = pd.DataFrame(
            [[elem[1] for elem in min]], index=[0], columns=[elem[0] for elem in min]
        )
        return pd.concat([mean_df, std_df, max_df, min_df], axis=1)
