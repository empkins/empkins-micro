from abc import abstractmethod

import pandas as pd
from tpcp import Algorithm, Parameter, make_action_safe


class BaseExtraction(Algorithm):
    """
    base class which defines the interface for all fiducial point extraction algorithms for ecg data
    xx: xx Pandas Dataframe containing the respiration Signal
    result: xx Saves resulting respiration rate as attribute in respiration_rate
    """

    _action_methods = "extract"

    # input parameters

    # results
    fiducial_points_positions_: pd.Series

    # constants

    #def __init__(self):

    # interface method (is implemented within each subclass)
    @abstractmethod
    @make_action_safe
    def extract(self, signal: pd.DataFrame, heart_beats_list: pd.DataFrame, sampling_rate_hz: float):
        pass



