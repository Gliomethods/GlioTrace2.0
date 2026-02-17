import numpy as np


def format_data(trajectories, columns, softmax_cols):
    """
    Convert a per-row trajectories DataFrame into per-trajectory arrays for GLM-HMM style training.

    Parameters
    ----------
    trajectories : pandas.DataFrame
        Long-format table containing all track points. Must include at least:
          - identifiers: `exp`, `roi`, `cellID`
          - temporal ordering: `time`
          - feature columns specified by `columns`
          - CNN probability columns specified by `softmax_cols`
    columns : sequence[str]
        Feature column names to keep (the returned X matrices will contain exactly these columns,
        in this order).
    softmax_cols : sequence[str]
        Column names containing per-class CNN outputs (assumed to be probabilities or scores that
        can be normalized row-wise).

    Returns
    -------
    trajectories_list : dict[tuple, np.ndarray]
        Mapping (exp, roi, cellID) -> X_feat of shape (T, F)
        Mapping (exp, roi, cellID) -> log-probabilities of shape (T, K)

    @ Author: André Lasses Armatowski
    """
    drop_cols = np.setdiff1d(trajectories.columns, columns)

    trajectories_list = {}   # key -> (T, F)
    cnn_outputs_list = {}    # key -> (T, K)

    for (exp, roi, cell_id), df_cell in trajectories.groupby(["exp", "roi", "cellID"], sort=False):
        key = (exp, roi, cell_id)
        df_cell = df_cell.sort_values("time")

        X_feat = df_cell.drop(columns=drop_cols).to_numpy(dtype=float)

        p = df_cell[softmax_cols].to_numpy(dtype=float)
        p = np.maximum(p, 1e-12)
        p = p / p.sum(axis=1, keepdims=True)

        out = np.log(p)  # (T, K)

        trajectories_list[key] = X_feat
        cnn_outputs_list[key] = out

    return trajectories_list, cnn_outputs_list


def add_universal_time_to_gammas(gamma_df, data_feat,
                                 keys=["exp", "roi", "cellID"], time_col="time",
                                 t_col="t"):
    """
    gamma_df must contain keys + t_col, where t runs 0..T_i-1 per key.
    Adds gamma_df[time_col] = gamma_df[t_col] + offset,
    where offset = min(data_feat[time_col]) for that key.

    @ Author: André Lasses Armatowski
    """

    # offsets per (exp, roi, cellID)
    offsets = (
        data_feat.groupby(list(keys), as_index=False)[time_col]
        .min()
        .rename(columns={time_col: "offset"})
    )

    # bring offsets onto gammas
    out = gamma_df.merge(offsets, on=list(
        keys), how="left", validate="many_to_one")

    out[time_col] = out[t_col].astype(int) + out["offset"].astype(int)
    out = out.drop(columns=["offset"])
    return out
