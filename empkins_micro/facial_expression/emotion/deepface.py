import datetime
from copy import deepcopy
from typing import Optional, Union, Sequence, Dict, Any

import cv2
import numpy as np
import pandas as pd
from biopsykit.utils._types import path_t
from deepface import DeepFace

from tqdm.auto import tqdm

from empkins_micro.facial_expression.emotion._base import _BaseEmotionProcessor

__all__ = ["DeepFaceEmotionProcessor"]


class DeepFaceEmotionProcessor(_BaseEmotionProcessor):
    def __init__(
        self, file_path: path_t, output_dir: Optional[path_t] = None, emotions: Optional[Sequence[str]] = None, **kwargs
    ):
        super().__init__(file_path, output_dir, emotions)
        self.raw_result: Dict[Any, Any] = {}
        self.cap = None
        self.detector_backend = kwargs.get("detector_backend", "ssd")

    def process(self, fps_out: Optional[float] = 1, start_time: Optional[Union[datetime.datetime, str]] = None):
        super().process()
        # open video
        self.cap = cv2.VideoCapture(str(self.file_path))
        # get fps from video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # analyze video
        frequency = int(fps / fps_out)
        frame_steps = np.arange(0, length, frequency)

        result_dict = {}
        if start_time:
            # ensure datetime
            start_time = pd.to_datetime(start_time)
            frame_index = (start_time.to_datetime64().astype(float) + (frame_steps / fps) * 1e9).astype(int)
            frame_index = pd.to_datetime(frame_index)
        else:
            frame_index = frame_steps / fps

        dict_keys = {"emotion": {}, "dominant_emotion": None, "region": {}}

        for frame_count, frame_idx in tqdm(list(zip(frame_steps, frame_index)), unit="frames"):
            out = deepcopy(dict_keys)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = self.cap.read()
            if not ret:  # end of video
                break

            result = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False, detector_backend=self.detector_backend
            )
            out.update(**result)
            result_dict[frame_idx] = out

        self.cap.release()

        emotion_result = pd.DataFrame(
            {key: {**val["emotion"], "dominant_emotion": val["dominant_emotion"]} for key, val in result_dict.items()}
        ).T
        emotion_result.columns.name = "emotion"
        emotion_result.index.name = "time"

        dominant_emotion = pd.DataFrame()
        dominant_emotion["emotion"] = emotion_result[["dominant_emotion"]].copy()
        dominant_emotion["emotion_numeric"] = dominant_emotion.replace(self.emotions_dict)

        self.raw_result = result_dict
        self.dominant_emotion = dominant_emotion
        self.emotion_data = emotion_result[self.emotions].copy()
