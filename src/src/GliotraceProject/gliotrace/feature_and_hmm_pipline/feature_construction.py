import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree

def w_dir_roi_t(data, power=2):
    """
    Author: Linnea Hallin
    """
    # calculates the weighted sin and cos of cells in a roi
    # (excluding the cells' own directions)
    # data: one roi and one time step
    # power: weights are calculated as 1/distance**power
    # returns: w_sin (weighted sin) and w_cos(weighted cos)
    num_cells = data.shape[0]
    distances = distance_matrix(data[["trax", "tray"]], data[["trax", "tray"]])
    w_sin = np.zeros(num_cells)
    w_cos = np.zeros(num_cells)
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # temporarily disables errors caused by division with zero
        for cell in range(num_cells):
            weights = 1 / (distances[cell,] ** power)
            weights[cell] = 0
            sin_nom = data["sin"] * weights
            cos_nom = data["cos"] * weights
            if np.isnan(sin_nom).all():
                w_sin[cell] = np.nan
                w_cos[cell] = np.nan
            else:
                sin = np.nansum(sin_nom) / np.nansum(weights)
                cos = np.nansum(cos_nom) / np.nansum(weights)
                sincosnorm = np.sqrt(sin**2 + cos**2)
                w_sin[cell] = sin / sincosnorm
                w_cos[cell] = cos / sincosnorm

    return pd.DataFrame(
        {"index": data["index"], "weighted_sin": w_sin, "weighted_cos": w_cos}
    )

def weighted_direction(data, power=2):
    """
    Author: Linnea Hallin
    """
    dir_sorted = (
        data.groupby(["roi", "time"], sort=False)
        .apply(lambda x: w_dir_roi_t(x, power=power), include_groups=False)
        .reset_index(["roi", "time"], drop=True)
    )
    dir_orig_order = dir_sorted.sort_values("index").drop("index", axis=1)
    return dir_orig_order

def feature_construction(data):
    """
    Author: Linnea Hallin, André Lasses Armatowski
    """

    # sort data (if data is already sorted this is O(n))
    data = data.sort_values(by=["roi", "cellID", "time"]).reset_index(drop=True)

    # Already in data:
    # sumgreen per ROI and time

    data["index"] = range(data.shape[0])

    # Only use microglia colocalization from tme label
    data["tme_label"] = data["tme_label"].replace({1: 2, 2: 2, 3: 1})

    # Number of cells in ROI:
    data["num_cells"] = data.groupby(["roi", "time"])["time"].transform("size")

    # Local growth-rate as delta-sumgreen:
    data["growth_rate"] = data.groupby(
        ["roi"])["sum_green"].diff() / data["delta_t"]

    # Velocity:
    data["dx"] = data.groupby(["roi", "cellID"])["trax"].diff()
    data["dy"] = data.groupby(["roi", "cellID"])["tray"].diff()

    # Direction and speed: Cell-wise:
    data["speed"] = np.sqrt(data["dx"] ** 2 + data["dy"]
                            ** 2) / data["delta_t"]
    data["direction"] = np.arctan2(data["dx"], data["dy"])

    # --------- velocity - ROI  ----------------

    # Roi speed
    data["roi_speed"] = data.groupby(["roi", "time"])[
        "speed"].transform("mean")

    # ROI direction:
    data["sin"] = np.sin(data["direction"])
    data["cos"] = np.cos(data["direction"])
    mean_sin = data.groupby(["roi", "time"])["sin"].transform("mean")
    mean_cos = data.groupby(["roi", "time"])["cos"].transform("mean")
    sincos_norm = np.sqrt(mean_sin**2 + mean_cos**2)
    data["roi_sin"] = mean_sin / sincos_norm
    data["roi_cos"] = mean_cos / sincos_norm
    data["roi_direction"] = np.arctan2(mean_sin, mean_cos)

    # Weighted direction
    w_dir = weighted_direction(data)
    data["weighted_sin"] = w_dir["weighted_sin"]
    data["weighted_cos"] = w_dir["weighted_cos"]
    # data = pd.concat([data, weighted_direction(data)], axis=1)

    # Measures the degree of sameness in angles 0 = uniform angles, 1 = paralell
    data["polarization"] = np.sqrt(mean_sin**2 + mean_cos**2)

    # ------------- velocity similarity with cell and cells in ROI- -------
    data["cos_sim"] = np.cos(data["roi_direction"] - data["direction"])
    data["faster"] = data["speed"] - data["roi_speed"]

    # ------------- Velocity change in time -------

    # Cell:
    data["directional_change"] = data.groupby(
        ["roi", "cellID"])["direction"].diff()
    data["speed_change"] = data.groupby(["roi", "cellID"])["speed"].diff()

    # Roi:
    data["roi_directional_change"] = data.groupby(["roi", "cellID"])[
        "roi_direction"
    ].diff()
    data["roi_speed_change"] = data.groupby(["roi", "cellID"])[
        "roi_speed"].diff()

    # --- Treatment interaction terms (mixture / varying-effect style) ---
    treat_ind = data["is_treatment"].astype(int)

    treat_cols = [
        "speed",
        "direction",
        "roi_speed",
        "roi_direction",
        "polarization",
        "cos_sim",
        "faster",
        "directional_change",
        "speed_change",
        "roi_directional_change",
        "roi_speed_change",
        "growth_rate",
        "sum_green",
        "tme_label"
    ]

    for col in treat_cols:
        if col in data.columns:
            data[f"{col}_treat"] = data[col] * treat_ind

    # Vascular distance is conditional, so handle separately
    if "vascular_distance" in data.columns:
        data["vascular_distance_treat"] = data["vascular_distance"] * treat_ind

    data.drop("index", axis=1)

    return data
