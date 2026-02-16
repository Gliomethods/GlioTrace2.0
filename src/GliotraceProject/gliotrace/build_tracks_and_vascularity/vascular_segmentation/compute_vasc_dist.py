import numpy as np
from scipy.spatial import cKDTree


def get_vasc_coords(vasc_mask):
    """
    Author: Linnea Hallin
    """
    ys, xs = np.nonzero(vasc_mask)
    return np.column_stack((xs, ys)).astype(float) + 1.0  # 1-indexed


def add_vascular_distance(df, binary_stack, p=0.2):
    """
    Author: Linnea Hallin
    """
    # binary_stack expected shape: (H, W, T)
    vasc_sum = np.sum(binary_stack, axis=2)
    th = p * np.max(df["time"])
    vasc_mask = vasc_sum >= th

    vasc_coords = get_vasc_coords(vasc_mask)
    tree = cKDTree(vasc_coords)

    dists, _ = tree.query(df[["trax", "tray"]].to_numpy(), k=1)
    df = df.copy()
    df["vascular_distance"] = dists
    return df
