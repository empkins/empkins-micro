import datetime
from typing import Optional, Union, Sequence

import pandas as pd
from biopsykit.utils._types import path_t

from empkins_micro.facial_expression.emotion._base import _BaseEmotionProcessor

try:
    from fer import Video, FER  # pylint:disable=import-outside-toplevel
except ImportError as e:
    raise ImportError(
        "'fer' is not installed that is required for facial expression recognition. "
        "Please install it with 'pip install fer' or 'poetry add fer'."
    ) from e

__all__ = ["FerEmotionProcessor"]


class FerEmotionProcessor(_BaseEmotionProcessor):
    def __init__(
        self, file_path: path_t, output_dir: Optional[path_t] = None, emotions: Optional[Sequence[str]] = None
    ):

        super().__init__(file_path, output_dir, emotions)
        self.raw_result = None

        self.video = Video(self.file_path, outdir=self.output_dir)
        # create facial detector object
        self.detector = FER()

    def process(self, fps_out: Optional[float] = 1, start_time: Optional[Union[datetime.datetime, str]] = None):
        super().process()
        # ensure datetime
        start_time = pd.to_datetime(start_time)
        # analyze video
        # raw_data = self.video.analyze(self.detector, save_fps=fs_out, save_frames=True, display=False)
        raw_data = self.video.analyze(self.detector, max_results=10, frequency=30, save_frames=True, display=False)
        self.raw_result = raw_data.copy()
        emotion_data = self.video.to_pandas(raw_data)
        emotion_data.index = emotion_data.index / fps_out
        time_index = (start_time.to_datetime64().astype(float) + emotion_data.index * 1e9).astype(int)
        emotion_data.index = pd.to_datetime(time_index)
        emotion_data.index.name = "time"
        emotion_data = emotion_data.drop(columns="box")
        emotion_data = emotion_data[self.emotions]
        self.emotion_data = emotion_data

    def dominant_emotion(self, data: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Sequence[str]]:
        if data is None:
            data = self.emotion_data
        dominant_emotion = data.idxmax(axis=1)
        dominant_emotion = dominant_emotion.replace(self.emotions_dict)
        dominant_emotion = pd.DataFrame(dominant_emotion, columns=["emotion"])
        return dominant_emotion
