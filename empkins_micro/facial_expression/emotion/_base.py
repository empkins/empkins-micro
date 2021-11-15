from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils._types import path_t


class _BaseEmotionProcessor(ABC):
    def __init__(
        self, file_path: path_t, output_dir: Optional[path_t] = None, emotions: Optional[Sequence[str]] = None
    ):
        file_path = Path(file_path)
        # ensure str
        if output_dir is None:
            output_dir = file_path.parent.joinpath("output")
        self.file_path = str(file_path)
        self.output_dir = str(output_dir)

        if emotions is None:
            emotions = ["happy", "surprise", "neutral", "sad", "angry", "disgust", "fear"]
        self.emotions = emotions

        self.emotion_data: Optional[pd.DataFrame] = None
        self.dominant_emotion: Optional[pd.DataFrame] = None

    @property
    def emotions(self) -> Sequence[str]:
        return self._emotions

    @emotions.setter
    def emotions(self, emotions: Sequence[str]):
        self._emotions = emotions
        self.emotions_dict = {emotion: i for i, emotion in enumerate(self._emotions)}

    @abstractmethod
    def process(self):
        return
