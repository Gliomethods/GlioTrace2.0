from gliotrace.feature_and_hmm_pipline.hmm_shrunken import update_emission_shrunken


def initalize_centroids(data, emb, gammas, keep_pct):
    """
    Author: André Lasses Armatowski, Linnea Hallin

    """

    pi = data["state_label"].value_counts(
        normalize=True).sort_index().to_numpy()

    # Get inital shrunken emission parameters
    Xs = list(emb.values())  # list of arrays (T_i, D)

    # Create initial centroids
    means, sd, active_mask = update_emission_shrunken(
        Xs,
        gammas,
        keep_pct=keep_pct
    )

    return means, sd, active_mask, pi
