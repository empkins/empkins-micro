import re
from pathlib import Path
from typing import Dict

import pandas as pd

from biopsykit.io import load_time_log
from biopsykit.utils._types import path_t


def process_time_log(path: path_t) -> pd.DataFrame:
    # ensure pathlib
    path = Path(path)
    subject_id = re.findall(r"time_log_(VP_\w+).csv", path.name)[0]
    # read timelog file
    timelog = pd.read_csv(path)
    # sort phases ascending
    timelog = timelog.sort_values("Von")
    # set index, drop unneeded columns, rename columns and index
    timelog = timelog.set_index("Aktivitätstyp")
    timelog = timelog.drop(
        index=["Speichelprobe", "Speichel", "Speichel Post"], columns=["Dauer", "Kommentar"], errors="ignore"
    )
    timelog.columns = ["start", "end"]
    timelog.index.rename(None, inplace=True)
    timelog = timelog.rename(index={"phy. Baseline": "pre_baseline"})
    timelog = timelog.rename(index={"Start": "pre"})

    # convert all phases to lower case, remove spaces and replace "phasen" (e.g. in "mist1_phasen") by "math"
    timelog.index = timelog.index.str.lower().str.strip().str.replace(" ", "_").str.replace("phasen", "math")

    timelog = pd.DataFrame(timelog.T.unstack(), columns=pd.Index([subject_id], name="subject"))
    timelog.loc[("post", "start"), :] = timelog.loc[("mist3_feedback", "end"), :]
    timelog.loc[("post", "end"), :] = timelog.loc[("pre", "end"), :]
    timelog.loc[("pre", "end"), :] = timelog.loc[("mist1_baseline", "start"), :]
    timelog = timelog.T

    return timelog


def load_split_time_logs(path: path_t) -> Dict[str, pd.DataFrame]:
    timelog_path = list(path.glob("*.csv"))[0]
    timelog = load_time_log(timelog_path, continuous_time=False)
    timelog_dict = {key: timelog.filter(like=key) for key in ["pre", "mist", "post"]}
    return timelog_dict
