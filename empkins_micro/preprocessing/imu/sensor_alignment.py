"""Helpers to rotate sensor data.

Copied and modified from *gaitmap - The Gait and Movement Analysis Package* (MaD Lab, FAU).
The original code can be found at: https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap.
"""

from typing import Optional, Union, Dict

import numpy as np
import pandas as pd

from biopsykit.signals.imu.static_moment_detection import _find_static_samples, METRIC_FUNCTION_NAMES
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype

from empkins_micro.preprocessing.imu.rotations import rotate_dataset, get_gravity_rotation

#: The gravity vector in m/s^2
GRAV_VEC = np.array([0.0, 0.0, 9.81])


def align_dataset_to_gravity(
    data: pd.DataFrame,
    sampling_rate_hz: float,
    window_length_s: Optional[float] = 0.7,
    static_signal_th: Optional[float] = 2.5,
    metric: Optional[METRIC_FUNCTION_NAMES] = "median",
    gravity: Optional[np.ndarray] = GRAV_VEC,
) -> pd.DataFrame:
    """Align dataset, so that each sensor z-axis (if multiple present in dataset) will be parallel to gravity.

    Median accelerometer vector will be extracted form static windows which will be classified by a sliding window
    with (window_length -1) overlap and a thresholding of the gyro signal norm. This will be performed for each sensor
    in the dataset individually.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)
    sampling_rate_hz : float
        Samplingrate of input signal in units of hertz.
    window_length_s : float
        Length of desired window in units of seconds.
    static_signal_th : float
       Threshold to decide whether a window should be considered as static or active. Window will be classified on
       <= threshold on gyro norm
    metric : str
        Metric which will be calculated per window, one of the following strings

        'maximum' (default)
            Calculates maximum value per window
        'mean'
            Calculates mean value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    gravity : vector with shape (3,), axis ([x, y ,z]), optional
        Expected direction of gravity during rest after the rotation.
        For example if this is `[0, 0, 1]` the sensor will measure +g on the z-axis after rotation (z-axis pointing
        upwards)

    Returns
    -------
    :class:`~pandas.DataFrame`
        This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    >>> # pd.DataFrame containing one or multiple sensor data streams, each of containing all 6 IMU
    ... # axis (acc_x, ..., gyr_z)
    >>> dataset_df = ...
    >>> align_dataset_to_gravity(dataset_df, window_length_s = 0.7, static_signal_th = 2.0, metric = 'median',
    ... gravity = np.array([0.0, 0.0, 1.0])
    <copy of dataset with all axis aligned to gravity>

    See Also
    --------
    :func:`biopsykit.signals.imu.static_moment_detection.find_static_moments` : Details on the used static moment
    detection function for this method.

    """
    _assert_is_dtype(data, pd.DataFrame)

    window_length = int(round(window_length_s * sampling_rate_hz))
    acc_vector: Union[np.ndarray, Dict[str, np.ndarray]]
    # get static acc vector
    acc_vector = _get_static_acc_vector(data, window_length, static_signal_th, metric)
    # get rotation to gravity
    rotation = get_gravity_rotation(acc_vector, gravity)

    return rotate_dataset(data, rotation)


def _get_static_acc_vector(
    data: pd.DataFrame, window_length: int, static_signal_th: float, metric: METRIC_FUNCTION_NAMES = "median"
) -> np.ndarray:
    """Extract the mean accelerometer vector describing the static position of the sensor."""
    # find static windows within the gyro data
    static_bool_array = _find_static_samples(
        data.filter(like="gyr").to_numpy(), window_length, static_signal_th, metric
    )

    # raise exception if no static windows could be found with given user settings
    if not any(static_bool_array):
        raise ValueError(
            "No static windows could be found to extract sensor offset orientation. Please check your input data or try"
            " to adapt parameters like window_length, static_signal_th or used metric."
        )

    # get mean acc vector indicating the sensor offset orientation from gravity from static sequences
    return np.median(data.filter(like="acc").to_numpy()[static_bool_array], axis=0)
