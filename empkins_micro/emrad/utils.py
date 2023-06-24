from typing import List, Union, Dict

import numpy as np
import pandas as pd


def input_conversion(
    x: Union[List, np.ndarray],
    input_type: str,
    output_type: str,
    sfreq: int = 1000,
) -> np.ndarray:
    """Convert input time series to the desired output format.

    This function is called by functions to convert time series to a different format.
    The input and outputs can be:
    * `peaks`: a boolean vector where `1` denote the detection of an event in the
    time-series.
    * `peaks_idx`: a 1d NumPy array of integers where each item is the sample index
    of an event in the time series.
    * `rr_ms`: a 1d NumPy array (integers or floats) of RR /peak-to-peak intervals
    in milliseconds.
    * `rr_s`: a 1d NumPy array (integers or floats) of RR /peak-to-peak intervals
    in seconds.

    Parameters
    ----------
    x :
        The input time series.
    input_type :
        The type of input provided (can be `"peaks"`, `"peaks_idx"`, `"rr_ms"`,
        `"rr_s"`).
    output_type :
        The type of desired output (can be `"peaks"`, `"peaks_idx"`, `"rr_ms"`,
        `"rr_s"`).
    sfreq :
        The sampling frequency (default is 1000 Hz). Only applies when `iput_type` is
        `"peaks"` or `"peaks_idx"`.

    Returns
    -------
    output :
        The time series converted to the desired format.

    """

    if output_type not in ["peaks", "peaks_idx", "rr_ms", "rr_s"]:
        raise ValueError("Invalid output type.")

    if input_type == output_type:
        raise ValueError("Input type and output type are the same.")

    x = np.asarray(x)

    if input_type == "peaks":
        if ((x == 1) | (x == 0)).all():
            if output_type == "rr_ms":
                output = (np.diff(np.where(x)[0]) / sfreq) * 1000
            elif output_type == "rr_s":
                output = np.diff(np.where(x)[0]) / sfreq
            elif output_type == "peaks_idx":
                output = np.where(x)[0]  # type: ignore
        else:
            raise ValueError("The peaks vector should only contain boolean values.")

    elif input_type == "peaks_idx":
        if (np.diff(x) > 0).all() & (np.rint(x) == x).all():
            if output_type == "rr_ms":
                output = (np.diff(x) / sfreq) * 1000
            elif output_type == "rr_s":
                output = np.diff(x) / sfreq
            elif output_type == "peaks":
                output = np.zeros(x[-1] + 1, dtype=bool)
                output[x] = True
        else:
            raise ValueError("Invalid peaks index provided.")

    elif input_type == "rr_ms":
        if (x > 0).all():
            if output_type == "rr_s":
                output = x / 1000
            elif output_type == "peaks":
                output = np.zeros(int(np.sum(x)) + 1, dtype=bool)
                output[np.cumsum(x)] = True
                output[0] = True
            elif output_type == "peaks_idx":
                output = np.cumsum(x)
                output = np.insert(output, 0, 0)
        else:
            raise ValueError("Invalid intervals provided.")

    elif input_type == "rr_s":
        if (x > 0).all():
            if output_type == "rr_ms":
                output = x * 1000  # type: ignore
            elif output_type == "peaks":
                output = np.zeros(np.sum(x * 1000).astype(int) + 1, dtype=bool)
                output[np.cumsum(x * 1000).astype(int)] = True
                output[0] = True
            elif output_type == "peaks_idx":
                output = np.cumsum(x * 1000).astype(int)
                output = np.insert(output, 0, 0)
        else:
            raise ValueError("Invalid intervals provided.")

    else:
        raise ValueError("Invalid input type.")

    return output


def correct_peaks(
    peaks: Union[List, np.ndarray],
    input_type: str = "peaks",
    extra_correction: bool = True,
    missed_correction: bool = True,
    n_iterations: int = 1,
    verbose: bool = False,
) -> Dict[str, Union[int, np.ndarray]]:
    """Correct long, short, extra, missed and ectopic beats in peaks vector.

    Parameters
    ----------
    peaks :
        Boolean vector of peaks.
    input_type :
        The type of input vector. Defaults to `"rr_ms"` for vectors of RR intervals, or
        interbeat intervals (IBI), expressed in milliseconds. Can also be a boolean
        vector where `1` represents the occurrence of R waves or systolic peakspeaks
        vector `"rr_s"` or IBI expressed in seconds.
    extra_correction :
      If `True` (default), correct extra peaks in the peaks time series.
    missed_correction :
      If `True` (default), correct missed peaks in the peaks time series.
    n_iterations :
        How many time to repeat the detection-correction process. Defaults to `1`.
    verbose :
        Control the verbosity of the function. Defaults to `True`.

    Returns
    -------
    correction :
        The corrected RR time series and the number of artefacts corrected:
        - clean_peaks: The corrected boolean time-serie.
        - extra: The number of extra beats corrected.
        - missed: The number of missed beats corrected.

    See also
    --------
    correct_rr

    Notes
    -----
    This function wil operate at the `peaks` vector level to keep the length of the
    signal constant after peaks correction.

    """

    peaks = np.asarray(peaks)

    if input_type != "peaks":
        peaks = input_conversion(peaks, input_type, output_type="peaks")

    clean_peaks = peaks.copy()
    nExtra, nMissed = 0, 0

    if verbose:
        print(f"Cleaning the peaks vector using {n_iterations} iterations.")

    for n_it in range(n_iterations):
        if verbose:
            print(f" - Iteration {n_it+1} - ")

        # Correct extra peaks
        if extra_correction:
            # Artefact detection
            artefacts = rr_artefacts(clean_peaks, input_type="peaks")

            if np.any(artefacts["extra"]):
                peaks_idx = np.where(clean_peaks)[0][1:]

                # Convert the RR interval idx to sample idx
                extra_idx = peaks_idx[np.where(artefacts["extra"])[0]]

                # Number of extra peaks to correct
                this_nextra = int(artefacts["extra"].sum())
                if verbose:
                    print(f"... correcting {this_nextra} extra peak(s).")

                nExtra += this_nextra

                # Removing peak n+1 to correct RR interval n
                clean_peaks[extra_idx] = False

                artefacts = rr_artefacts(clean_peaks, input_type="peaks")

        # Correct missed peaks
        if missed_correction:
            if np.any(artefacts["missed"]):
                peaks_idx = np.where(clean_peaks)[0][1:]

                # Convert the RR interval idx to sample idx
                missed_idx = peaks_idx[np.where(artefacts["missed"])[0]]

                # Number of missed peaks to correct
                this_nmissed = int(artefacts["missed"].sum())
                if verbose:
                    print(f"... correcting {this_nmissed} missed peak(s).")

                nMissed += this_nmissed

                # Correct missed peaks using sample index
                for this_idx in missed_idx:
                    clean_peaks = correct_missed_peaks(clean_peaks, this_idx)

                # Artefact detection
                artefacts = rr_artefacts(clean_peaks, input_type="peaks")

    return {
        "clean_peaks": clean_peaks,
        "extra": nExtra,
        "missed": nMissed,
    }


def correct_missed_peaks(peaks: Union[List, np.ndarray], idx: int) -> np.ndarray:
    """Correct missed peaks in boolean peak vector.

    The new peak is placed in the middle of the previous interval.

    Parameters
    ----------
    peaks :
        Boolean vector of peaks detection.
    idx : int
        Index of the peaks corresponding to the missed RR interval. The new peaks will
        be placed between this one and the previous one.

    Returns
    -------
    clean_peaks :
        Corrected boolean vector of peaks.
    """
    peaks = np.asarray(peaks, dtype=bool)

    if not peaks[idx]:
        raise (ValueError("The index provided does not match with a peaks."))

    clean_peaks = peaks.copy()

    # The index of the previous peak
    previous_idx = np.where(clean_peaks[:idx])[0][-1]

    # Estimate new interval
    interval = int((idx - previous_idx) / 2)

    # Add peak in vector
    clean_peaks[previous_idx + interval] = True

    return clean_peaks


def rr_artefacts(
    rr: Union[List, np.ndarray],
    c1: float = 0.13,
    c2: float = 0.17,
    alpha: float = 5.2,
    input_type: str = "rr_ms",
) -> Dict[str, np.ndarray]:
    """Artefacts detection from RR time series using the subspaces approach
    proposed by Lipponen & Tarvainen (2019).

    Parameters
    ----------
    rr :
        1d numpy array of RR intervals (in seconds or miliseconds) or peaks
        vector (boolean array).
    c1 :
        Fixed variable controling the slope of the threshold lines. Default is
        `0.13`.
    c2 :
        Fixed variable controling the intersect of the threshold lines. Default
        is `0.17`.
    alpha :
        Scaling factor used to normalize the RR intervals first deviation.
    input_type :
        The type of input vector. Defaults to `"rr_ms"` for vectors of RR
        intervals, or  interbeat intervals (IBI), expressed in milliseconds.
        Can also be a boolean vector where `1` represents the occurrence of
        R waves or systolic peakspeaks vector `"rr_s"` or IBI expressed in
        seconds.

    Returns
    -------
    artefacts :
        Dictionary storing the parameters of RR artefacts rejection. All the vectors
        outputed have the same length as the provided RR time serie:

        * subspace1 : np.ndarray
            The first dimension. First derivative of R-R interval time serie.
        * subspace2 : np.ndarray
            The second dimension (1st plot).
        * subspace3 : np.ndarray
            The third dimension (2nd plot).
        * mRR : np.ndarray
            The mRR time serie.
        * ectopic : np.ndarray
            Boolean array indexing probable ectopic beats.
        * long : np.ndarray
            Boolean array indexing long RR intervals.
        * short : np.ndarray
            Boolean array indexing short RR intervals.
        * missed : np.ndarray
            Boolean array indexing missed RR intervals.
        * extra : np.ndarray
            Boolean array indexing extra RR intervals.
        * threshold1 : np.ndarray
            Threshold 1.
        * threshold2 : np.ndarray
            Threshold 2.

    Notes
    -----
    This function will use the method proposed by [1]_ to detect ectopic beats, long,
    shorts, missed and extra RR intervals.

    Examples
    --------
    >>> from systole import simulate_rr
    >>> from systole.detection import rr_artefacts
    >>> rr = simulate_rr()  # Simulate RR time series
    >>> artefacts = rr_artefacts(rr)
    >>> print(artefacts.keys())
    dict_keys(['subspace1', 'subspace2', 'subspace3', 'mRR', 'ectopic', 'long',
    'short', 'missed', 'extra', 'threshold1', 'threshold2'])

    References
    ----------
    .. [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173-181. https://doi.org/10.1080/03091902.2019.1640306

    """
    rr = np.asarray(rr)

    if input_type != "rr_ms":
        rr = input_conversion(rr, input_type, output_type="rr_ms")

    ###########
    # Detection
    ###########

    # Subspace 1 (dRRs time serie)
    dRR = np.diff(rr, prepend=0)
    dRR[0] = dRR[1:].mean()  # Set first item to a realistic value

    dRR_df = pd.DataFrame({"signal": np.abs(dRR)})
    q1 = dRR_df.rolling(91, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = dRR_df.rolling(91, center=True, min_periods=1).quantile(0.75).signal.to_numpy()

    th1 = alpha * ((q3 - q1) / 2)
    dRR = dRR / th1
    s11 = dRR

    # mRRs time serie
    medRR = (
        pd.DataFrame({"signal": rr})
        .rolling(11, center=True, min_periods=1)
        .median()
        .signal.to_numpy()
    )
    mRR = rr - medRR
    mRR[mRR < 0] = 2 * mRR[mRR < 0]

    mRR_df = pd.DataFrame({"signal": np.abs(mRR)})
    q1 = mRR_df.rolling(91, center=True, min_periods=1).quantile(0.25).signal.to_numpy()
    q3 = mRR_df.rolling(91, center=True, min_periods=1).quantile(0.75).signal.to_numpy()

    th2 = alpha * ((q3 - q1) / 2)
    mRR /= th2

    # Subspace 2
    ma = np.hstack(
        [0, [np.max([dRR[i - 1], dRR[i + 1]]) for i in range(1, len(dRR) - 1)], 0]
    )
    mi = np.hstack(
        [0, [np.min([dRR[i - 1], dRR[i + 1]]) for i in range(1, len(dRR) - 1)], 0]
    )
    s12 = ma
    s12[dRR < 0] = mi[dRR < 0]

    # Subspace 3
    ma = np.hstack(
        [[np.max([dRR[i + 1], dRR[i + 2]]) for i in range(0, len(dRR) - 2)], 0, 0]
    )
    mi = np.hstack(
        [[np.min([dRR[i + 1], dRR[i + 2]]) for i in range(0, len(dRR) - 2)], 0, 0]
    )
    s22 = ma
    s22[dRR >= 0] = mi[dRR >= 0]

    ##########
    # Decision
    ##########

    # Find ectobeats
    cond1 = (s11 > 1) & (s12 < (-c1 * s11 - c2))
    cond2 = (s11 < -1) & (s12 > (-c1 * s11 + c2))
    ectopic = cond1 | cond2
    # No ectopic detection and correction at time serie edges
    ectopic[-2:] = False
    ectopic[:2] = False

    # Find long or shorts
    longBeats = ((s11 > 1) & (s22 < -1)) | ((np.abs(mRR) > 3) & (rr > np.median(rr)))
    shortBeats = ((s11 < -1) & (s22 > 1)) | ((np.abs(mRR) > 3) & (rr <= np.median(rr)))

    # Test if next interval is also outlier
    for cond in [longBeats, shortBeats]:
        for i in range(len(cond) - 2):
            if cond[i] is True:
                if np.abs(s11[i + 1]) < np.abs(s11[i + 2]):
                    cond[i + 1] = True

    # Ectopic beats are not considered as short or long
    shortBeats[ectopic] = False
    longBeats[ectopic] = False

    # Missed vector
    missed = np.abs((rr / 2) - medRR) < th2
    missed = missed & longBeats
    longBeats[missed] = False  # Missed beats are not considered as long

    # Etra vector
    extra = np.abs(rr + np.append(rr[1:], 0) - medRR) < th2
    extra = extra & shortBeats
    shortBeats[extra] = False  # Extra beats are not considered as short

    # No short or long intervals at time serie edges
    shortBeats[0], shortBeats[-1] = False, False
    longBeats[0], longBeats[-1] = False, False

    artefacts = {
        "subspace1": s11,
        "subspace2": s12,
        "subspace3": s22,
        "mRR": mRR,
        "ectopic": ectopic,
        "long": longBeats,
        "short": shortBeats,
        "missed": missed,
        "extra": extra,
        "threshold1": th1,
        "threshold2": th2,
    }

    return artefacts
