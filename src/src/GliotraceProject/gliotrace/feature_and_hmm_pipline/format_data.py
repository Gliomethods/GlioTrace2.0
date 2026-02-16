import numpy as np


def format_data(trajectories, columns, softmax_cols):
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
