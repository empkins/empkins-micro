import pandas as pd
import numpy as np
from pathlib import Path
from empkins_micro.utils._types import path_t
from typing import Optional


class DerivedFeatureExtraction:
    _base_path: path_t
    _subject_id: str
    _condition: str
    _diarization: pd.DataFrame

    def __init__(
            self,
            base_path: path_t,
            subject_id: str,
            condition: str,
            diarization: Optional[pd.DataFrame] = None,

    ):

        self._base_path = base_path
        self._subject_id = subject_id
        self._condition = condition
        self._diarization = diarization
