from gliotrace.visualize.vis_tracking import vis_tracking_morphology_from_rows

from pathlib import Path
import numpy as np


def generate_video(
    stacktable,
    track_data,
    exp,
    roi,
    channel_roles={"blue": "none", "green": "gbm", "red": "vasc"},
    output=None,
):
    roi_data = track_data.loc[(track_data["exp"] == exp) & (
        track_data["roi"] == roi)].copy()

    # Order channels, can also work to check if a channel is present
    for key, ch in channel_roles.items():
        if ch == "gbm":
            gbm_channel = key
        if ch == "vasc":
            vasc_channel = key

    mask = (stacktable["exp"] == exp) & (stacktable["roi"] == roi)

    rows = stacktable.loc[mask, "file_path"]

    if rows.empty:
        raise ValueError(f"No stack found for exp={exp}, roi={roi}")

    if len(rows) > 1:
        raise ValueError(f"Multiple stacks found for exp={exp}, roi={roi}")

    stack_path = Path(rows.iloc[0])

    data = np.load(stack_path, allow_pickle=True)

    # Map channels
    channel_data = {
        "blue": data["Bstack"],
        "green": data["Tstack"],
        "red": data["Vstack"],
    }

    gbm = channel_data[gbm_channel].astype(
        np.uint8) if gbm_channel is not None else None
    vasc = channel_data[vasc_channel].astype(
        np.uint8) if vasc_channel is not None else None

    # Enforce divisibility by 8 in dimensions to avoid problems in vascular segmentation
    new_pixel_size = int(np.floor(gbm.shape[0] / 8) * 8)
    gbm = gbm[0:new_pixel_size, 0:new_pixel_size, :]
    vasc = vasc[0:new_pixel_size, 0:new_pixel_size, :]

    # ---------------- Visualization ------------------
    output_str = vis_tracking_morphology_from_rows(
        roi_data,
        gbm,
        vasc,
        output,
    )

    return output_str
