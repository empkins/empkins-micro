from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from empkins_micro.utils._types import path_t

__all__ = ["SpeakerDiarization"]


class SpeakerDiarization:
    """Class to perform speaker diarization on audio files."""

    _audio_path: path_t
    _pyannote_auth_token: str
    _audio_data: pd.DataFrame
    _sampling_rate_hz: int
    _min_segment_length_sec: float

    def __init__(
        self,
        audio_path: path_t,
        pyannote_auth_token: str,
        min_segment_length_sec: Optional[float] = 0.3,
    ):
        """Initialize speaker diarization class.

        Parameters
        ----------
        audio_path : path_t
            Path to audio file.
        pyannote_auth_token : str
            Authentication token for speaker diarization using 'pyannote'.
        min_segment_length_sec : float, optional
            Minimum length of speaker segments in seconds in order to be detected. Default: 0.3

        """
        self._audio_path = Path(audio_path)
        self._pyannote_auth_token = pyannote_auth_token
        self._min_segment_length_sec = min_segment_length_sec
        self._audio_data, self._sampling_rate_hz = self._load_audio_file()

    @property
    def audio_data(self) -> pd.DataFrame:
        """Return audio data as pandas DataFrame.

        Returns
        -------
        :class:`pandas.DataFrame`
            Audio data as pandas DataFrame.

        """
        return self._audio_data

    def _load_audio_file(self) -> Tuple[pd.DataFrame, int]:
        fs, data = wavfile.read(self._audio_path)
        data = pd.DataFrame(data)
        data.index = pd.Index(np.around(data.index / fs, decimals=10), name="t")
        return data, fs

    def _speaker_diarization_pyannote(self) -> pd.DataFrame:
        """Perform speaker diarization using 'pyannote'.

        Returns
        -------
        :class:`pandas.DataFrame`
            Raw speaker diarization data as pandas DataFrame.
        """
        try:
            from pyannote.audio import Pipeline
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Module 'pyannote.audio' not found. Please install it manually or "
                "install 'empkins-micro' with 'audio' extras via "
                "'poetry install empkins_micro -E audio'"
            ) from e
        # pyannote pipeline for speaker diarization
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self._pyannote_auth_token
        )
        pipeline.to(torch.device("mps"))
        diarization = pipeline(self._audio_path)  # apply pipeline to audio file

        speaker_data = [
            {"start": turn.start, "stop": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]  # df for storing diarization

        speaker_data = pd.DataFrame(speaker_data)
        return speaker_data

    @staticmethod
    def _clean_diarization_output(
        speaker_data: pd.DataFrame, min_segment_length: float
    ) -> pd.DataFrame:
        """Clean speaker diarization output.

        Cleans the speaker diarization output by removing segments that are shorter than the minimum segment length.

        Parameters
        ----------
        speaker_data : :class:`pd.DataFrame`
            Speaker diarization data.
        min_segment_length : float
            Minimum length of speaker segments in seconds in order to be detected.

        Returns
        -------
        :class:`pd.DataFrame`

        """
        data = speaker_data.assign(
            **{"length": speaker_data["stop"] - speaker_data["start"]}
        )

        indices = np.where(data["length"] < min_segment_length)[
            0
        ]  # identify segments smaller than the given minimum length
        data = data.drop(index=indices, axis=0).reset_index(
            drop=True
        )  # drop identified segments
        return data

    @staticmethod
    def _identify_test_speaker(speaker_data: pd.DataFrame) -> str:
        data = speaker_data.set_index("speaker")
        data = data[["length"]].groupby("speaker").sum()
        return data.idxmax()[0]

    # def save_diarization(self):
    #    file_name = self.audio_path.stem  # original file name, used for saving the diarization as file
    # df.to_csv(file_name"_diarization" + ".txt")

    def _speaker_one_hot_encoding(
        self, speaker_data: pd.DataFrame, test_speaker: str
    ) -> pd.DataFrame:
        speaker = np.unique(speaker_data["speaker"])  # array with speaker names
        # speaker_panel includes all speakers which are not the test_speaker
        speaker_panel = speaker[speaker != test_speaker]

        columns = list(speaker) + ["SPEAKER_PANEL"]
        # initialize binary speaker dataframe
        speaker_one_hot = pd.DataFrame(index=self.audio_data.index, columns=columns)

        # translate diarization segments to binary speaker vectors
        for i, row in speaker_data.iterrows():
            current_speaker = row["speaker"]
            speaker_one_hot.loc[row["start"] : row["stop"], current_speaker] = 1
            speaker_one_hot.loc[row["start"] : row["stop"], "SPEAKER_PANEL"] = (
                current_speaker in speaker_panel
            )

        speaker_one_hot = speaker_one_hot.fillna(0).astype(int)
        return speaker_one_hot

    @staticmethod
    def _clean_speaker_overlap(
        speaker_one_hot: pd.DataFrame, test_speaker: str
    ) -> pd.DataFrame:
        # if 2 or more speakers are detected at the same time -> set test_speaker to zero
        data = speaker_one_hot.copy()
        data.loc[speaker_one_hot.sum(axis=1) > 1, test_speaker] = 0
        return data

    @staticmethod
    def _update_speaker_segments(
        speaker_data: pd.DataFrame, speaker_one_hot: pd.DataFrame, test_speaker: str
    ):
        data_new_segments = []
        # identify test_speaker segments
        speaker_list = [test_speaker, "SPEAKER_PANEL_INV"]
        speaker_one_hot["SPEAKER_PANEL_INV"].iloc[[0, -1]] = 0
        for speaker in speaker_list:
            subject_diff = speaker_one_hot[speaker].diff()
            start = speaker_one_hot.loc[subject_diff == 1].index
            stop = speaker_one_hot.loc[subject_diff == -1].index
            data = pd.concat([pd.DataFrame(start), pd.DataFrame(stop)], axis=1)
            data.columns = ["start", "stop"]
            # add speaker column
            data["speaker"] = speaker
            data_new_segments.append(data)

        data_new_segments = data_new_segments + [
            speaker_data[speaker_data["speaker"] != test_speaker]
        ]
        data_new_segments = pd.concat(data_new_segments)
        data_new_segments = data_new_segments.assign(
            **{"length": data_new_segments["stop"] - data_new_segments["start"]}
        )
        data_new_segments = data_new_segments.sort_values(by="start").reset_index(
            drop=True
        )
        return data_new_segments

    @staticmethod
    def _apply_speaker_encoding(data: pd.DataFrame, test_speaker: str) -> pd.DataFrame:
        if data[test_speaker].sum() == 0 and data["SPEAKER_PANEL"].sum() != 0:
            data.loc[:, "SPEAKER_PANEL"] = 1
        return data

    def _group_diarization(
        self, speaker_one_hot: pd.DataFrame, test_speaker: str
    ) -> pd.DataFrame:
        speaker_one_hot = speaker_one_hot.assign(
            **{
                "group_id": speaker_one_hot[[test_speaker]]
                .diff()
                .fillna(0)
                .astype(int)
                .abs()
                .cumsum()
            }
        )

        speaker_one_hot = speaker_one_hot.groupby("group_id", group_keys=False).apply(
            lambda df: self._apply_speaker_encoding(df, test_speaker)
        )
        speaker_one_hot["SPEAKER_PANEL_INV"] = (
            1 - speaker_one_hot["SPEAKER_PANEL"]
        )  # group test subject segments

        return speaker_one_hot

    def speaker_diarization(self) -> pd.DataFrame:
        speaker_data = self._speaker_diarization_pyannote()
        speaker_data = self._clean_diarization_output(
            speaker_data, min_segment_length=self._min_segment_length_sec
        )
        test_speaker = self._identify_test_speaker(speaker_data)
        speaker_one_hot_data = self._speaker_one_hot_encoding(
            speaker_data, test_speaker
        )
        speaker_one_hot_data = self._clean_speaker_overlap(
            speaker_one_hot_data, test_speaker
        )
        speaker_one_hot_data = self._group_diarization(
            speaker_one_hot_data, test_speaker
        )
        speaker_data = self._update_speaker_segments(
            speaker_data, speaker_one_hot_data, test_speaker
        )
        speaker_data.index.name = "segment_id"
        return speaker_data
