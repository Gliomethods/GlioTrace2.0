import pandas as pd
import numpy as np


def feature_construction(data):
    """
    @ Author: Linnea Hallin, André Lasses Armatowski
    """

    data["index"] = range(data.shape[0])

    # Only use microglia colocalization from tme label
    data["tme_label"] = data["tme_label"].replace({1: 1, 2: 0, 3: 0})

    # Number of cells in ROI:
    data["num_cells"] = data.groupby(["roi", "time"])["time"].transform("size")

    # Velocity:
    dx = data.groupby(["roi", "cellID"])["trax"].diff()
    dy = data.groupby(["roi", "cellID"])["tray"].diff()

    # Direction and speed: Cell-wise:
    data["speed"] = np.sqrt(dx ** 2 + dy ** 2) / data["delta_t"]
    data["direction"] = np.arctan2(dx, dy)

    # ROI direction:
    data["sin"] = np.sin(data["direction"])
    data["cos"] = np.cos(data["direction"])
    mean_sin = data.groupby(["roi", "time"])["sin"].transform("mean")
    mean_cos = data.groupby(["roi", "time"])["cos"].transform("mean")
    data["roi_direction"] = np.arctan2(mean_sin, mean_cos)

    # Measures the degree of sameness in angles 0 = uniform angles, 1 = paralell
    data["polarization"] = np.sqrt(mean_sin**2 + mean_cos**2)
    data.loc[data["num_cells"] == 1, "polarization"] = 0.5

    # Cosine similarity
    data["cos_sim"] = np.cos(data["roi_direction"] - data["direction"])

    # Rescale vascular_distance
    vd = data["vascular_distance"].to_numpy(dtype=np.float32)
    fs = data["frame_size"].to_numpy(dtype=np.float32)

    # replace NaN/Inf with frame_size
    bad = ~np.isfinite(vd)
    vd[bad] = fs[bad]

    # log-transform
    data["vascular_distance"] = np.log1p(vd)

    # --- Treatment interaction terms (mixture / varying-effect style) ---
    treat_ind = data["is_treatment"].astype(int)

    treat_cols = [
        "speed",
        "direction",
        "polarization",
        "cos_sim",
        "tme_label",
        "sum_green",
        "vascular_distance"
    ]

    for col in treat_cols:
        if col in data.columns:
            data[f"{col}_treat"] = data[col] * treat_ind

    data.drop("index", axis=1)

    return data
