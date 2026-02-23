# General statistics
from gliotrace.build_tracks_and_vascularity.general_statistics.gbm_stats import compute_gbm_stats

# Classification and track cells
from gliotrace.build_tracks_and_vascularity.classify_and_track.track_classify import track_classify

# Segmentation of vascularity
from gliotrace.build_tracks_and_vascularity.vascular_segmentation.vascular_segmentation import compute_vascular_segementation

# Compute vascular distance
from gliotrace.build_tracks_and_vascularity.vascular_segmentation.compute_vasc_dist import add_vascular_distance


def build_tracks_and_vascularity(
    gbm,
    vasc,
    gbm_net,
    tme_net,
    seg_net,
    blocksize,
    detection_sensitivity,
    i,
    dt
):
    """
        Build cell tracks with per-frame classification features and vasculature-derived distances.

        Parameters
        ----------
        gbm : np.ndarray
            GBM (cell) image stack used for detection/tracking/classification 
        vasc : np.ndarray
            Vasculature image stack used for vessel segmentation and distance-to-vessel computation
        gbm_net : torch.nn.Module
            Trained "state" network used inside `track_classify` to assign cell state probabilities/labels.
        tme_net : torch.nn.Module
            Trained TME (binary) network used inside `track_classify` to compute TME-related outputs.
        seg_net : torch.nn.Module
            Trained segmentation network used to segment vasculature from `vasc`.
        blocksize : int
            Spatial block size / tiling parameter forwarded to `track_classify` (pipeline-specific).
        detection_sensitivity : float
            Detection sensitivity parameter forwarded to `track_classify`.
        i : int
            Index/identifier forwarded to `track_classify`
        dt : float
            Time step between frames, forwarded to `compute_gbm_stats` for temporal feature computation.

        Returns
        -------
        tracks_with_props : pandas.DataFrame or None
            Table of tracked objects augmented with:
            - `sum_green`: per-frame "sum green" statistic mapped from `compute_gbm_stats`
            - `adMAD`: per-frame adMAD statistic mapped from `compute_gbm_stats`
            - vascular distance features added by `add_vascular_distance`
            Returns None if tracking/classification produced no tracks.

    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """
    # ------------------------------------------
    # Tracking and classifying
    # ------------------------------------------
    _, _, adMAD, sg = compute_gbm_stats(gbm, dt)

    # Track and classify the cells
    tracks_with_props = track_classify(
        gbm,
        vasc,
        detection_sensitivity,
        gbm_net,
        tme_net,
        blocksize,
        i,
    )

    if tracks_with_props is None:
        return None

    # Append sum green to the times
    sum_green_map = dict(enumerate(sg, start=0))
    tracks_with_props["sum_green"] = tracks_with_props["time"].map(
        sum_green_map
    )

    # Append adMAD to the times
    adMAD_map = dict(enumerate(adMAD, start=1))  # time 1..T-1
    tracks_with_props["adMAD"] = tracks_with_props["time"].map(
        adMAD_map)

    # --------------------------------------------
    # Vascular segmentation for vascular distance
    # --------------------------------------------
    _, _, binary_stack = compute_vascular_segementation(
        vasc, net=seg_net
    )

    tracks_with_props = add_vascular_distance(tracks_with_props, binary_stack)

    return tracks_with_props
