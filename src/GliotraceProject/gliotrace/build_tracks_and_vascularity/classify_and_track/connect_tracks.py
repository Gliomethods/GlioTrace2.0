import numpy as np


def connect_tracklets(traX, traY, phenotypes, max_radius=15.0):
    """
    Iterate over frames and merge track segments whose endpoints lie within an
    iteratively increasing matching distance. When a stop point of an older
    track matches the start point of a newer track, the newer track is appended
    to the older one.

    Parameters
    ----------
    traX : numpy.ndarray
        Matrix of x-coordinates for all tracks, shaped (num_frames, num_tracks).
        Each column represents a track over time.
    traY : numpy.ndarray
        Matrix of y-coordinates for all tracks, shaped (num_frames, num_tracks).
    phenotypes : numpy.ndarray or list of numpy.ndarray
        Frame-wise properties of each tracked cell.

    Returns
    -------
    traX : numpy.ndarray
        Updated x-coordinate track matrix after merging connected tracks.
    traY : numpy.ndarray
        Updated y-coordinate track matrix after merging connected tracks.
    phenotypes : same type as input
        Adjusted phenotype matrices reflecting the merged trajectories.

    @ Author: André Lasses Armatowski, Madeleine Skeppås, Sven Nelander
    """

    traX = traX.copy()
    traY = traY.copy()
    phenotypes = [p.copy() for p in phenotypes]

    T, N = traX.shape

    for i in range(2, T):
        stops_mask = np.isnan(traX[i, :]) & ~np.isnan(traX[i - 1, :])
        track_stops = np.where(stops_mask)[0]

        starts_mask = ~np.isnan(traX[i, :]) & np.isnan(traX[i - 1, :])
        track_starts = np.where(starts_mask)[0]

        if track_stops.size == 0 or track_starts.size == 0:
            continue

        # Coordinates of stops at frame i-1
        p_x = traX[i - 1, track_stops]  # (n_stops,)
        p_y = traY[i - 1, track_stops]  # (n_stops,)

        # Coordinates of starts at frame i
        q_x = traX[i, track_starts]  # (n_starts,)
        q_y = traY[i, track_starts]  # (n_starts,)

        # Pairwise distances between starts (q) and stops (p)
        # shape: (n_starts, n_stops)
        dx = q_x[:, None] - p_x[None, :]
        dy = q_y[:, None] - p_y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)

        # For each start, find nearest stop
        nearest_stop_idx = np.argmin(dist, axis=1)  # (n_starts,)
        nearest_dist = dist[np.arange(dist.shape[0]), nearest_stop_idx]

        # Only keep those within max_radius
        valid = nearest_dist <= max_radius
        if not np.any(valid):
            continue

        # Candidates: (distance, start_local_idx, stop_local_idx)
        cand_start_local = np.where(valid)[0]
        cand_stop_local = nearest_stop_idx[valid]
        cand_dists = nearest_dist[valid]

        order = np.argsort(cand_dists)
        used_stops = set()
        used_starts = set()
        matched_start_local = []
        matched_stop_local = []

        for idx in order:
            s_loc = cand_start_local[idx]
            p_loc = cand_stop_local[idx]
            if (s_loc in used_starts) or (p_loc in used_stops):
                continue
            used_starts.add(s_loc)
            used_stops.add(p_loc)
            matched_start_local.append(s_loc)
            matched_stop_local.append(p_loc)

        if len(matched_start_local) == 0:
            continue

        matched_start_local = np.array(matched_start_local, dtype=int)
        matched_stop_local = np.array(matched_stop_local, dtype=int)

        # Global column indices
        start_cols = track_starts[matched_start_local]
        stop_cols = track_stops[matched_stop_local]

        traX[i:, stop_cols] = traX[i:, start_cols]
        traY[i:, stop_cols] = traY[i:, start_cols]
        traX[i:, start_cols] = np.nan
        traY[i:, start_cols] = np.nan

        # Do the same for phenotypes
        for idx, data in enumerate(phenotypes):
            data[i:, stop_cols] = data[i:, start_cols]
            data[i:, start_cols] = np.nan
            phenotypes[idx] = data

    # Remove tracks that are entirely NaN (erased by stitching)
    keep = ~np.all(np.isnan(traX), axis=0)
    traX = traX[:, keep]
    traY = traY[:, keep]
    phenotypes = [p[:, keep] for p in phenotypes]

    return traX, traY, phenotypes
