import numpy as np
import pandas as pd

def binarize_eyeblink(data):

    if len(data.index) != 0:
        num_frames = np.floor(data.at[0, "vid_dur"] * data.at[0, "fps"]).astype(int)
        bin_data = np.zeros((num_frames, 1))
        indices = data["mov_blinkframes"].to_numpy()
        indices = indices[indices <= num_frames]
        bin_data[indices] = 1

        return pd.DataFrame(bin_data.astype(int), columns=["mov_eyeblink"])

    return pd.DataFrame(columns=["mov_eyeblink"])