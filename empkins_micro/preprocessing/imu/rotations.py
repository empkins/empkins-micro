"""A set of util functions that ease handling rotations.

All util functions use :class:`scipy.spatial.transform.Rotation` to represent rotations.

Copied and modified from *gaitmap - The Gait and Movement Analysis Package* (MaD Lab, FAU).
The original code can be found at: https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform.rotation import Rotation

#: The gravity vector in m/s^2 in the FSF
from empkins_micro.utils.vector_math import find_orthogonal, normalize, row_wise_dot

GRAV_VEC = np.array([0.0, 0.0, 9.81])


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis ([x, y ,z]) or array of rotation axis
    angle : float or array with shape (n,)
        rotation angle or array of angeles in rad

    Returns
    -------
    rotation(s) : :class:`~scipy.spatial.transform.rotation.Rotation`
        Rotation object with len n

    Examples
    --------
    Single rotation: 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
    >>> rot.as_quat().round(decimals=3)
    array([1., 0., 0., 0.])
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -0., -1.],
           [ 0., -1.,  0.]])

    Multiple rotations: 90 and 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad([90, 180]))
    >>> rot.as_quat().round(decimals=3)
    array([[0.707, 0.   , 0.   , 0.707],
           [1.   , 0.   , 0.   , 0.   ]])
    >>> # In case of multiple rotations, the first rotation is applied to the first vector
    >>> # and the second to the second
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -1.,  0.],
           [ 0., -1.,  0.]])

    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))


def get_gravity_rotation(gravity_vector: np.ndarray, expected_gravity: np.ndarray = GRAV_VEC) -> Rotation:
    """Find the rotation matrix needed to align  z-axis with gravity.

    Parameters
    ----------
    gravity_vector : vector with shape (3,)
        axis ([x, y ,z])
    expected_gravity : vector with shape (3,)
        axis ([x, y ,z])

    Returns
    -------
    rotation : :class:`~scipy.spatial.transform.rotation.Rotation`
        rotation between given gravity vector and the expected gravity

    Examples
    --------
    >>> goal = np.array([0, 0, 1])
    >>> start = np.array([1, 0, 0])
    >>> rot = get_gravity_rotation(start)
    >>> rotated = rot.apply(start)
    >>> rotated
    array([0., 0., 1.])

    """
    gravity_vector = normalize(gravity_vector)
    expected_gravity = normalize(expected_gravity)
    return find_shortest_rotation(gravity_vector, expected_gravity)


def find_shortest_rotation(v1: np.ndarray, v2: np.ndarray) -> Rotation:
    """Find a quaternion that rotates v1 into v2 via the shortest way.

    Parameters
    ----------
    v1 : vector with shape (3,)
        axis ([x, y ,z])
    v2 : vector with shape (3,)
        axis ([x, y ,z])

    Returns
    -------
    rotation : :class:`~scipy.spatial.transform.rotation.Rotation`
        Shortest rotation that rotates v1 into v2

    Examples
    --------
    >>> goal = np.array([0, 0, 1])
    >>> start = np.array([1, 0, 0])
    >>> rot = find_shortest_rotation(start, goal)
    >>> rotated = rot.apply(start)
    >>> rotated
    array([0., 0., 1.])

    """
    if (not np.isclose(np.linalg.norm(v1, axis=-1), 1)) or (not np.isclose(np.linalg.norm(v2, axis=-1), 1)):
        raise ValueError("v1 and v2 must be normalized")
    axis = find_orthogonal(v1, v2)
    angle = find_unsigned_3d_angle(v1, v2)
    return rotation_from_angle(axis, angle)


def rotate_dataset(data: pd.DataFrame, rotation: Rotation) -> pd.DataFrame:
    """Apply a rotation to acc and gyro data of a dataset.

    Parameters
    ----------
    data : :class`~pandas.DataFrame`
        Dataframe containing data from one sensor
    rotation : :class:`~scipy.spatial.transform.rotation.Rotation`
        Rotation that will be applied to the sensor data

    Returns
    -------
    rotated data : :class:`~pandas.DataFrame`
        Rotated data. This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    This will apply the one rotation

    >>> data = ...  # dataframe with sensor data
    >>> rotate_dataset(data, rotation=rotation_from_angle(np.array([0, 0, 1]), np.pi))
    <copy of dataset with all axis rotated>

    """
    return _rotate_sensor(data, rotation, inplace=False)


def _rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    gyro_cols = data.filter(like="gyr").columns
    acc_cols = data.filter(like="acc").columns
    data[gyro_cols] = rotation.apply(data[gyro_cols].to_numpy())
    data[acc_cols] = rotation.apply(data[acc_cols].to_numpy())
    return data


def find_unsigned_3d_angle(v1: np.ndarray, v2: np.ndarray) -> Union[np.ndarray, float]:
    """Find the angle (in rad) between two 3D vectors.

    Parameters
    ----------
    v1 : vector with shape (3,)  or array of vectors
        axis ([x, y ,z]) or array of axis
    v2 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis

    Returns
    -------
        angle or array of angles between two vectors

    Examples
    --------
    two vectors: 1D

    >>> find_unsigned_3d_angle(np.array([-1, 0, 0]), np.array([-1, 0, 0]))
    0

    two vectors: 2D

    >>> find_unsigned_3d_angle(np.array([[-1, 0, 0],[-1, 0, 0]]), np.array([[-1, 0, 0],[-1, 0, 0]]))
    array([0,0])

    """
    v1_, v2_ = np.atleast_2d(v1, v2)
    v1_ = normalize(v1_)
    v2_ = normalize(v2_)
    out = np.arccos(row_wise_dot(v1_, v2_) / (np.linalg.norm(v1_, axis=-1) * np.linalg.norm(v2_, axis=-1)))
    if v1.ndim == 1:
        return np.squeeze(out)
    return out
