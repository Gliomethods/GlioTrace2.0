from gliotrace.visualize.vis_tracking_compare import vis_tracking_morphology_compare_viterbi

from pathlib import Path
import numpy as np
import pandas as pd


def generate_video_compare(
    track_data,
    stacktable,
    data_feat,
    exp,
    roi,
    channel_roles={"blue": "none", "green": "gbm", "red": "vasc"},
    output=None,
):
    # --- Base ROI slices ---
    td_roi = track_data.loc[(track_data["exp"] == exp)
                            & (track_data["roi"] == roi)].copy()
    df_roi = data_feat.loc[(data_feat["exp"] == exp) &
                           (data_feat["roi"] == roi)].copy()

    # --- Join coordinates (trax/tray) from tracked data into data_feat ---
    # We keep labels from data_feat (state_label + viterbi_label/state) and coords from track_data
    # Join key: (exp, roi, cellID, time)
    coord_cols = ["exp", "roi", "cellID", "time", "trax", "tray"]
    missing = [c for c in coord_cols if c not in td_roi.columns]
    if missing:
        raise ValueError(
            f"track_data is missing required coordinate columns: {missing}")

    td_coords = td_roi[coord_cols].drop_duplicates(
        subset=["exp", "roi", "cellID", "time"])

    # Ensure merge dtypes line up
    for c in ["exp", "roi", "cellID", "time"]:
        if c in df_roi.columns:
            df_roi[c] = df_roi[c].astype(int)
        if c in td_coords.columns:
            td_coords[c] = td_coords[c].astype(int)

    roi_data = df_roi.merge(
        td_coords,
        on=["exp", "roi", "cellID", "time"],
        how="left",
        validate="m:1",  # many feature-rows per unique time? usually 1:1; relax if needed
    )

    # Optional: drop rows without coords (can't draw them anyway)
    roi_data = roi_data.dropna(subset=["trax", "tray"])

    # --- Order channels ---
    gbm_channel = None
    vasc_channel = None
    for key, ch in channel_roles.items():
        if ch == "gbm":
            gbm_channel = key
        if ch == "vasc":
            vasc_channel = key

    # --- Find stack ---
    mask = (stacktable["exp"] == exp) & (stacktable["roi"] == roi)
    rows = stacktable.loc[mask, "file_path"]

    if rows.empty:
        raise ValueError(f"No stack found for exp={exp}, roi={roi}")
    if len(rows) > 1:
        raise ValueError(f"Multiple stacks found for exp={exp}, roi={roi}")

    stack_path = Path(rows.iloc[0])
    data = np.load(stack_path, allow_pickle=True)

    # --- Map channels ---
    channel_data = {
        "blue": data["Bstack"],
        "green": data["Tstack"],
        "red": data["Vstack"],
    }

    gbm = channel_data[gbm_channel].astype(
        np.uint8) if gbm_channel is not None else None
    vasc = channel_data[vasc_channel].astype(
        np.uint8) if vasc_channel is not None else None

    # Enforce divisibility by 8 to avoid problems in vascular segmentation
    new_pixel_size = int(np.floor(gbm.shape[0] / 8) * 8)
    gbm = gbm[0:new_pixel_size, 0:new_pixel_size, :]
    vasc = vasc[0:new_pixel_size, 0:new_pixel_size, :]

    # ---------------- Visualization ------------------
    output_str = vis_tracking_morphology_compare_viterbi(
        roi_data,
        gbm,
        vasc,
        output,
    )

    return output_str
