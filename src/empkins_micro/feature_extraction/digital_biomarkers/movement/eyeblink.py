import numpy as np
import pandas as pd


def _empty_eyeblink(error_text):
    data = {
        "mov_eyeblink": [np.nan],
        "error": [error_text],
    }
    return pd.DataFrame.from_dict(data)


def binarize_eyeblink(data):

    try:
        if len(data.index) != 0:
            num_frames = np.floor(data.at[0, "vid_dur"] * data.at[0, "fps"]).astype(int)
            bin_data = np.zeros((num_frames, 1))
            indices = data["mov_blinkframes"].to_numpy()
            indices = indices[indices <= num_frames]
            bin_data[indices] = 1

            df_eyeblink = pd.DataFrame(bin_data.astype(int), columns=["mov_eyeblink"])
            df_eyeblink["error"] = "PASS"

            return df_eyeblink

        return _empty_eyeblink("no blinks registered in this video")
    except Exception as e:
        return _empty_eyeblink(f"binarizing the eyeblinks failed: {e}")
