import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def get_vasc_coords(vasc_mask):
    """
    Author: Linnea Hallin
    """
    ys, xs = np.nonzero(vasc_mask)
    return np.column_stack((xs, ys)).astype(float) + 1.0  # 1-indexed

def vasc_dist_roi_t(data_roi, vasc_masks, time, window_size = 21, th=2):
    """
    Author: Linnea Hallin
    """
    if(window_size % 2 == 0):
        window_size += 1  # make sure window size is odd
        print("Warning: window_size should be odd, increasing by 1.")
    tmax = vasc_masks.shape[2]
    if(time < window_size//2):
        window = range(window_size)
    elif(time > tmax - window_size//2 - 1):
        window = range(tmax - window_size, tmax)
    else:
        window = range(time - int(window_size//2), time + int(window_size//2) + 1)
    vasc_sum = np.sum(vasc_masks[:,:,window], axis=2)
    vasc_mask = vasc_sum >= th
    idx = data_roi["time"] == time

    vasc_coords = get_vasc_coords(vasc_mask)
    tree = cKDTree(vasc_coords)  # nice structure for calculating distances
    dists, _ = tree.query(data_roi[["trax", "tray"]][idx], k=1)
    return dists

def vascular_distance(tracks, vasc_masks, window_size=21, th=2):
    """
    Author: Linnea Hallin
    """
    data = tracks.copy()
    frames = []
    data["index"] = range(len(data))  # to restore original order later
    for time in data["time"].unique():
        dists = vasc_dist_roi_t(
            data, vasc_masks, time, window_size=window_size, th=th
        )
        temp_df = pd.DataFrame(
            {"index": data.loc[data["time"] == time, "index"], "dist": dists}
        )
        frames.append(temp_df)
    dist = pd.concat(frames, ignore_index=False)
    dist = dist.sort_values("index").drop("index", axis=1)
    return dist

def add_vascular_distance(tracks, vasc_masks, window_size=21, th=2):
    """
    Author: Linnea Hallin
    """
    dists = vascular_distance(tracks, vasc_masks, window_size=window_size, th=th)
    tracks["vascular_distance"] = dists
    return tracks
