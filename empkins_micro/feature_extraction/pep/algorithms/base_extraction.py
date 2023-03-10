from abc import abstractmethod

import pandas as pd
from tpcp import Algorithm, Parameter, make_action_safe
from typing import List


class BaseExtraction(Algorithm):
    """base class which defines the interface for all fiducial point extraction algorithms

    results:
        points_ : saves positions of extracted points in pd.DataFrame
    """

    _action_methods = "extract"

    # results
    points_: pd.DataFrame

    # interface method
    @abstractmethod
    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """function which extracts specific fiducial points from cleaned signal, implementation within subclasses"""

        pass

    @staticmethod
    def match_points_heartbeats(points: List[int], heartbeats: pd.DataFrame) -> pd.DataFrame:
        """matches the given points to the corresponding heartbeats

        (such that returned DataFrame's format matches heartbeats df)

        Args:
            points: list of fiducial points of signal (samples)
            heartbeats: pd.DataFrame as returned by HeartBeatExtraction

        Returns:
            pd.DataFrame with point locations saved in row of associated heartbeat
        """

        points_heartbeats = pd.DataFrame(index=heartbeats.index, columns=["point_sample"])  # create result df

        # TODO evt check, ob länge gleich ist, ob punkt übrig ist, ob zB q-peak vor r liegt oder sowas (warnings für sanity checks)?
        for p in points:
            # find heartbeat to which point belongs and write point into corresponding row
            idx = heartbeats.loc[(heartbeats["start_sample"] < p) & (p < heartbeats["end_sample"])].index
            points_heartbeats["point_sample"].loc[idx] = p

        return points_heartbeats

