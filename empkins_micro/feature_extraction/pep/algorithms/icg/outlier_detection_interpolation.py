import pandas as pd
import numpy as np

from scipy.signal import argrelmin, butter, filtfilt
from scipy.stats import median_abs_deviation

from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from tpcp import Algorithm, Parameter, make_action_safe

from empkins_micro.feature_extraction.pep.algorithms.base_extraction import BaseExtraction

import warnings

class BPointExtractionForouzanfar(BaseExtraction):
    """algorithm to correct outliers based on [Forouzanfar et al., 2018, Psychophysiology]"""


    @make_action_safe
    def extract(self, b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int):
        """function which corrects outliers of given B-Point dataframe

        Args:
            b_points:
                pd.DataFrame containing the extracted B-Points per heartbeat, index functions as id of heartbeat
            c-points:
                pd.DataFrame containing the extracted C-Points per heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ICG signal in hz

        Returns:
            saves resulting corrected B-point locations (samples) in points_ attribute of super class,
            index is B-point (/heartbeat) id
        """
        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point"])


        self.points_ = corrected_b_points
        return self
