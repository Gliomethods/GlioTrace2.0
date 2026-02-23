import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def track_tumor_cells2(cellsx, cellsy, properties):
    """
    Kalman tracking.

    Parameters
    ----------
    cellsx : list of array-like
        Frame-wise x-coordinates of detected cells.
    cellsy : list of array-like
        Frame-wise y-coordinates of detected cells.
    properties : list of pandas.DataFrame
        Frame-wise tables of per-cell properties.

    Returns
    -------
    traX : numpy.ndarray
        (num_tracked_frames, num_tracks) matrix of x positions (measured when matched,
        predicted when missing-but-kept, NaN when terminated).
    traY : numpy.ndarray
        Same for y positions.
    x_hat_history : list of numpy.ndarray
        [x, vx, y, vy] histories, each (num_tracked_frames, num_tracks).
    phenotypes : list of numpy.ndarray
        List of phenotype histories aligned to tracks, each (num_tracked_frames, num_tracks).
        Here phenotypes are only ["cell_index", "frame_index"] like in your original.
    phenonames : list of str
        Names of phenotype columns.
    Kn_history : numpy.ndarray
        (2, num_frames) summary Kalman gain per frame:
          Kn_history[0, j] = mean K_x_from_xmeas across active tracks at frame j
          Kn_history[1, j] = mean K_vx_from_xmeas across active tracks at frame j
    startidx : int
        First frame index with non-empty properties (and used to initialize tracks).

    @ Author: André Lasses Armatowski, Madeleine Skeppås, Sven Nelander
    """

    # -------------------------
    # Model (constant velocity)
    # -------------------------
    dt = 1.0
    nstates = 4

    F = np.array(
        [
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    H = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )

    # Keep noise choices (identity)
    P0 = np.eye(nstates, dtype=float)
    Q = np.eye(nstates, dtype=float)
    R = np.eye(2, dtype=float)

    # Tracking params
    distance_threshold = 18.0
    max_missed_frames = 2

    # Assignment constants
    large_cost = 1e6

    # prefer real matches at exactly threshold
    dummy_cost = distance_threshold + 1e-6

    # -------------------------
    # Basic input checks
    # -------------------------
    n_frames = len(cellsx)
    if len(cellsy) != n_frames or len(properties) != n_frames:
        raise ValueError(
            "cellsx, cellsy, and properties must have the same length.")

    # -------------------------
    # Find first non-empty properties frame
    # -------------------------
    valid_indices = [i for i, p in enumerate(
        properties) if p is not None and not p.empty]
    if not valid_indices:
        raise ValueError("No non-empty properties found in any frame.")

    startidx = valid_indices[0]

    cols_to_keep = ["cell_index", "frame_index"]
    for c in cols_to_keep:
        if c not in properties[startidx].columns:
            raise KeyError(
                f"Missing required column '{c}' in properties at frame {startidx}.")

    # -------------------------
    # Initialize at startidx
    # -------------------------
    x0 = np.asarray(cellsx[startidx], dtype=float)
    y0 = np.asarray(cellsy[startidx], dtype=float)

    if x0.size == 0 or y0.size == 0:
        raise ValueError(
            f"Frame {startidx} has empty detections; cannot initialize tracks.")
    if x0.shape[0] != y0.shape[0]:
        raise ValueError(
            f"cellsx[{startidx}] and cellsy[{startidx}] must have same length.")

    props0 = properties[startidx][cols_to_keep].to_numpy()
    if props0.shape[0] != x0.shape[0]:
        raise ValueError(
            f"properties[{startidx}] row count ({props0.shape[0]}) "
            f"does not match detections ({x0.shape[0]})."
        )

    n_phenotypes = props0.shape[1]  # 2
    phenonames = cols_to_keep

    n_tracks = x0.shape[0]

    # Posterior state/cov at start frame (per track)
    # state = [x, vx, y, vy]
    x_post = np.vstack(
        [x0, np.zeros_like(x0), y0, np.zeros_like(y0)])  # (4, n_tracks)
    P_post = np.repeat(P0[np.newaxis, :, :], n_tracks,
                       axis=0)          # (n_tracks, 4, 4)

    # Outputs start at startidx
    traX = x0[np.newaxis, :]
    traY = y0[np.newaxis, :]

    x_hat_history = [
        x0[np.newaxis, :],                        # x
        np.zeros((1, n_tracks), dtype=float),     # vx
        y0[np.newaxis, :],                        # y
        np.zeros((1, n_tracks), dtype=float),     # vy
    ]

    phenotypes = [props0[:, i].astype(float)[np.newaxis, :]
                  for i in range(n_phenotypes)]
    Kn_history = np.zeros((2, n_frames), dtype=float)

    # Missed-frame counters
    track_counter = defaultdict(int)

    # Predict to next frame (prior for frame startidx+1)
    x_pred = F @ x_post
    P_pred = np.empty_like(P_post)
    for i in range(n_tracks):
        P_pred[i] = F @ P_post[i] @ F.T + Q

    I = np.eye(nstates, dtype=float)

    # -------------------------
    # Main loop
    # -------------------------
    for j in range(startidx + 1, n_frames):
        n_tracks = traX.shape[1]

        # Active tracks are those not terminated in previous output row
        active_tracks = np.where(~np.isnan(traX[-1, :]))[0]

        # Current detections
        xj = np.asarray(cellsx[j], dtype=float)
        yj = np.asarray(cellsy[j], dtype=float)
        if xj.shape[0] != yj.shape[0]:
            raise ValueError(
                f"cellsx[{j}] and cellsy[{j}] must have same length.")

        if xj.size == 0:
            props_j = np.zeros((0, n_phenotypes), dtype=float)
            current_pos = np.zeros((0, 2 + n_phenotypes), dtype=float)
        else:
            if properties[j] is not None and not properties[j].empty:
                for c in cols_to_keep:
                    if c not in properties[j].columns:
                        raise KeyError(
                            f"Missing required column '{c}' in properties at frame {j}.")
                props_j = properties[j][cols_to_keep].to_numpy()
                if props_j.shape[0] != xj.shape[0]:
                    raise ValueError(
                        f"properties[{j}] row count ({props_j.shape[0]}) "
                        f"does not match detections ({xj.shape[0]})."
                    )
            else:
                props_j = np.full((xj.size, n_phenotypes), np.nan, dtype=float)

            current_pos = np.column_stack([xj, yj, props_j]).astype(float)

        meas_positions = current_pos[:, 0:2]  # (n_meas, 2)
        n_meas = meas_positions.shape[0]

        # -------------------------
        # Assignment (active tracks only)
        # -------------------------
        cells_paired_prev = np.array([], dtype=int)  # track indices
        cells_paired_curr = np.array([], dtype=int)  # detection indices

        if active_tracks.size > 0 and n_meas > 0:
            pred_positions = x_pred[np.ix_(
                [0, 2], active_tracks)].T  # (n_active, 2)

            cost = cdist(pred_positions.astype(np.float32),
                         meas_positions.astype(np.float32))
            cost[~np.isfinite(cost)] = large_cost
            cost[cost > distance_threshold] = large_cost  # hard gate

            # Add dummy columns so each track can choose "no match"
            # cost_aug shape: (n_active, n_meas + n_active)
            cost_aug = np.hstack(
                [
                    cost,
                    np.full((active_tracks.size, active_tracks.size),
                            dummy_cost, dtype=np.float32),
                ]
            )

            row_ind, col_ind = linear_sum_assignment(cost_aug)

            mt, mc = [], []
            for r, c in zip(row_ind, col_ind):
                # real detection chosen and within gate
                if c < n_meas and cost[r, c] <= distance_threshold:
                    mt.append(active_tracks[r])
                    mc.append(c)

            if mt:
                cells_paired_prev = np.asarray(mt, dtype=int)
                cells_paired_curr = np.asarray(mc, dtype=int)

        # -------------------------
        # New tracks / old tracks
        # -------------------------
        all_current = np.arange(n_meas, dtype=int)
        newtracks = np.setdiff1d(all_current, cells_paired_curr)

        oldtracks = np.setdiff1d(active_tracks, cells_paired_prev)

        oldtracks_keep = []
        for t in oldtracks:
            track_counter[t] += 1
            if track_counter[t] <= max_missed_frames:
                oldtracks_keep.append(t)
        oldtracks_keep = np.asarray(oldtracks_keep, dtype=int)

        for t in cells_paired_prev:
            track_counter[t] = 0

        # -------------------------
        # Kalman update (per track)
        # -------------------------
        x_post_new = np.full((nstates, n_tracks), np.nan, dtype=float)
        P_post_new = np.full((n_tracks, nstates, nstates), np.nan, dtype=float)

        # Matched -> full update
        for t, c in zip(cells_paired_prev, cells_paired_curr):
            xp = x_pred[:, t]
            Pp = P_pred[t]
            z = meas_positions[c].astype(float)

            S = H @ Pp @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            K = Pp @ H.T @ S_inv
            innov = z - (H @ xp)

            x_upd = xp + K @ innov

            # Joseph form
            P_upd = (I - K @ H) @ Pp @ (I - K @ H).T + K @ R @ K.T
            P_upd = 0.5 * (P_upd + P_upd.T)

            x_post_new[:, t] = x_upd
            P_post_new[t] = P_upd

        # Unmatched but kept -> prediction-only (NO covariance update!)
        for t in oldtracks_keep:
            x_post_new[:, t] = x_pred[:, t]
            P_post_new[t] = P_pred[t]

        # -------------------------
        # Kalman gain history summary (mean over active tracks)
        # -------------------------
        k00, k10 = [], []
        for t in active_tracks:
            Pp = P_pred[t]
            if not np.isfinite(Pp).all():
                continue
            S = H @ Pp @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            Kt = Pp @ H.T @ S_inv
            k00.append(Kt[0, 0])
            k10.append(Kt[1, 0])

        Kn_history[0, j] = float(np.mean(k00)) if k00 else 0.0
        Kn_history[1, j] = float(np.mean(k10)) if k10 else 0.0

        # -------------------------
        # Expand output matrices for new tracks
        # -------------------------
        n_new = int(newtracks.size)
        if n_new > 0:
            traX = np.hstack(
                [traX, np.full((traX.shape[0], n_new), np.nan, dtype=float)])
            traY = np.hstack(
                [traY, np.full((traY.shape[0], n_new), np.nan, dtype=float)])

            for s in range(nstates):
                x_hat_history[s] = np.hstack(
                    [x_hat_history[s], np.full(
                        (x_hat_history[s].shape[0], n_new), np.nan, dtype=float)]
                )

            for p in range(n_phenotypes):
                phenotypes[p] = np.hstack(
                    [phenotypes[p], np.full(
                        (phenotypes[p].shape[0], n_new), np.nan, dtype=float)]
                )

        # -------------------------
        # Build output rows for this frame
        #   - matched: output measured x/y, store filtered state
        #   - missing-but-kept: output predicted x/y, phenotype NaN, store predicted state
        #   - new: output measured x/y, initialize state to [x,0,y,0] for this frame
        # -------------------------
        total_tracks = n_tracks + n_new

        row_x = np.full((total_tracks,), np.nan, dtype=float)
        row_y = np.full((total_tracks,), np.nan, dtype=float)
        row_ph = [np.full((total_tracks,), np.nan, dtype=float)
                  for _ in range(n_phenotypes)]
        row_state = [np.full((total_tracks,), np.nan, dtype=float)
                     for _ in range(nstates)]

        # matched
        for t, c in zip(cells_paired_prev, cells_paired_curr):
            row_x[t] = meas_positions[c, 0]
            row_y[t] = meas_positions[c, 1]
            for p in range(n_phenotypes):
                row_ph[p][t] = props_j[c, p]
            for s in range(nstates):
                row_state[s][t] = x_post_new[s, t]

        # missing but kept
        for t in oldtracks_keep:
            row_x[t] = x_post_new[0, t]
            row_y[t] = x_post_new[2, t]
            for s in range(nstates):
                row_state[s][t] = x_post_new[s, t]

        # new tracks appended as new columns
        if n_new > 0:
            new_cols = np.arange(n_tracks, n_tracks + n_new, dtype=int)
            for col_idx, det_idx in zip(new_cols, newtracks):
                row_x[col_idx] = meas_positions[det_idx, 0]
                row_y[col_idx] = meas_positions[det_idx, 1]
                for p in range(n_phenotypes):
                    row_ph[p][col_idx] = props_j[det_idx, p]

                # Initialize state estimate for birth frame (this is sane and consistent)
                row_state[0][col_idx] = meas_positions[det_idx, 0]
                row_state[1][col_idx] = 0.0
                row_state[2][col_idx] = meas_positions[det_idx, 1]
                row_state[3][col_idx] = 0.0

        # append rows to outputs
        traX = np.vstack([traX, row_x.reshape(1, -1)])
        traY = np.vstack([traY, row_y.reshape(1, -1)])

        for p in range(n_phenotypes):
            phenotypes[p] = np.vstack(
                [phenotypes[p], row_ph[p].reshape(1, -1)])

        for s in range(nstates):
            x_hat_history[s] = np.vstack(
                [x_hat_history[s], row_state[s].reshape(1, -1)])

        # -------------------------
        # Update posterior state arrays and predict to next frame
        # -------------------------
        if n_new > 0:
            x_new = np.vstack(
                [
                    meas_positions[newtracks, 0],
                    np.zeros(n_new, dtype=float),
                    meas_positions[newtracks, 1],
                    np.zeros(n_new, dtype=float),
                ]
            )
            P_new = np.repeat(P0[np.newaxis, :, :], n_new, axis=0)

            x_post = np.hstack([x_post_new, x_new])
            P_post = np.concatenate([P_post_new, P_new], axis=0)
        else:
            x_post = x_post_new
            P_post = P_post_new

        # Predict priors for next frame
        x_pred = F @ x_post
        P_pred = np.full_like(P_post, np.nan, dtype=float)
        for i in range(P_post.shape[0]):
            if np.isfinite(P_post[i]).all():
                P_pred[i] = F @ P_post[i] @ F.T + Q

    return traX, traY, x_hat_history, phenotypes, phenonames, Kn_history, startidx
