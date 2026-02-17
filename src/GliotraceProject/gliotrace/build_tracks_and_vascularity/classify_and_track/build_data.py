import pandas as pd
import numpy as np


def build_track_dataframes(
    traX,
    traY,
    phenotypes,
    phenonames,
    properties_meta,
    startidx,
):
    """
    Build per-track meta and embedding DataFrames from tracking and per-frame properties.

    Behavior:
    - Tracks are SPLIT whenever data is missing (x, y, frame_index, or cell_index is NaN).
    - When valid data resumes, a NEW cellID is created.
    - time is defined: time = startidx + t

    Returns
    -------
    tracks_with_props : pd.DataFrame

    @ Author: André Lasses Armatowski
    """

    # --- Identify phenotype matrix positions ---
    frame_index_pos = phenonames.index("frame_index")
    cell_index_pos = phenonames.index("cell_index")

    traX = np.asarray(traX, dtype=float)
    traY = np.asarray(traY, dtype=float)

    frame_index_mat = np.asarray(phenotypes[frame_index_pos], dtype=float)
    cell_index_mat = np.asarray(phenotypes[cell_index_pos], dtype=float)

    num_timepoints, num_tracks = traX.shape

    # --- Build lookup tables ---
    def build_lookup(dfs, name):
        lookup = {}
        if dfs is None:
            return lookup
        for df in dfs:
            if df is None or df.empty:
                continue
            if "frame_index" not in df.columns or "cell_index" not in df.columns:
                raise KeyError(
                    f"{name} DataFrames must contain frame_index and cell_index.")
            for row in df.to_dict(orient="records"):
                fi = row["frame_index"]
                ci = row["cell_index"]
                if pd.isna(fi) or pd.isna(ci):
                    continue
                lookup[(int(fi), int(ci))] = row
        return lookup

    meta_lookup = build_lookup(properties_meta, "properties_meta")

    # --- Build output rows ---
    rows_meta = []
    next_cell_id = 0

    for track_id in range(num_tracks):
        valid = (
            np.isfinite(traX[:, track_id])
            & np.isfinite(traY[:, track_id])
            & np.isfinite(frame_index_mat[:, track_id])
            & np.isfinite(cell_index_mat[:, track_id])
        )

        in_segment = False
        seg_cell_id = None

        for t in range(num_timepoints):
            if not valid[t]:
                in_segment = False
                seg_cell_id = None
                continue

            if not in_segment:
                seg_cell_id = next_cell_id
                next_cell_id += 1
                in_segment = True

            x = float(traX[t, track_id])
            y = float(traY[t, track_id])
            frame_index = int(frame_index_mat[t, track_id])
            cell_index = int(cell_index_mat[t, track_id])
            key = (frame_index, cell_index)

            time_val = startidx + t

            # --- Meta ---
            meta_row = meta_lookup.get(key)
            if meta_row is not None:
                entry = dict(meta_row)
                entry["trax"] = x
                entry["tray"] = y
                entry["cellID"] = seg_cell_id
                entry["time"] = time_val
                rows_meta.append(entry)

    tracks_with_props = pd.DataFrame(rows_meta)

    tracks_with_props = tracks_with_props.drop(
        columns=["frame_index", "cell_index"], errors="ignore"
    )

    if not tracks_with_props.empty:
        tracks_with_props = tracks_with_props.sort_values(
            ["cellID", "time"], ignore_index=True
        )

    return tracks_with_props
