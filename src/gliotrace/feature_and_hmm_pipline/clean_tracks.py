def filter_features_3(
    data_feat,
    fcols,
    hard_coded_features,
    min_timepoints=2,
    verbose=True,
):
    """
    Filter feature dataframe by:
      - removing rows with missing transition covariates (fcols),
      - removing tracks with too few timepoints.

    Returns:
      data_feat_filtered

    @ Author: André Lasses Armatowski
    """

    group_cols = ["exp", "roi", "cellID"]
    time_col = "time"
    keys = ["exp", "roi", "cellID", "time"]

    # ---------------- Basic key sanity ----------------
    missing_keys = [c for c in keys if c not in data_feat.columns]
    if missing_keys:
        raise KeyError(
            f"Missing required columns in data_feat: {missing_keys}")

    if data_feat.duplicated(keys).any():
        raise ValueError(
            "data_feat has duplicate (exp, roi, cellID, time) rows")

    # Sort so time is consistent within tracks
    data_feat = data_feat.sort_values(keys).copy()

    # ---------------- Column ordering ----------------
    full_cols = list(fcols) + \
        [c for c in hard_coded_features if c not in fcols]

    missing = [c for c in full_cols if c not in data_feat.columns]
    if missing:
        raise KeyError(f"Missing columns in data_feat: {missing}")

    data_feat = data_feat.loc[:, full_cols].copy()

    # ---------------- Filter away NA feature rows (only require fcols) ----------------
    mask_no_na = data_feat.loc[:, fcols].notna().all(axis=1)
    data_feat = data_feat.loc[mask_no_na].copy()

    # ---------------- Filter away tracks with too few timepoints ----------------
    sizes = data_feat.groupby(group_cols, sort=False)[
        time_col].transform("size")
    mask_big_enough = sizes.ge(min_timepoints)

    data_feat = data_feat.loc[mask_big_enough].copy()

    if data_feat.shape[0] == 0:
        if verbose:
            print("Not enough cells to determine transitions")
        return None

    data_feat = data_feat.reset_index(drop=True)
    return data_feat
