import numpy as np

from gliotrace.build_tracks_and_vascularity.classify_and_track.connect_tracks import connect_tracklets
from gliotrace.build_tracks_and_vascularity.classify_and_track.find_cell import macro_track2
from gliotrace.build_tracks_and_vascularity.classify_and_track.track_cell import track_tumor_cells2
from gliotrace.build_tracks_and_vascularity.classify_and_track.classify import classify_tumor_cells
from gliotrace.build_tracks_and_vascularity.classify_and_track.build_data import build_track_dataframes


def track_classify(
    gbm,
    vasc,
    detection_sensitivity,
    net_const,
    tme_net_const,
    blocksize,
    stack_index,
):
    """
    Compute cell-level statistics that require both the GBM (green channel) and
    vasculature (red channel) image stacks. The workflow performs detection,
    classification, tracking, and final assembly of track-wise properties.

    Parameters
    ----------
    gbm : numpy.ndarray
        gbm-channel stack containing tumor cell signal.
    vasc : numpy.ndarray
        vasc-channel stack containing vasculature signal.
    detection_sensitivity : float
        Sensitivity factor controlling LoG filtering and detection thresholds.
        Higher values make cell detection more permissive.
    net_const : dict
        Parameters and/or model handles for the morphology-classification network.
    tme_net_const : dict
        Parameters and/or model handles for the TME-interaction classification network.
    blocksize : int
        Patch size used when extracting cell snapshots for feature classification.
    stack_index : int
        Identifier of the current image stack (passed through to classifiers).

    Returns
    -------
    tracks_with_props : pandas.DataFrame
        Track-wise table containing cell positions, track IDs, time indices, and
        all per-frame meta properties.
    tracks_embeddings : pandas.DataFrame
        Track-wise table containing embedding vectors (emb_*) aligned to
        track/time pairs.

    Notes
    -----
    The function performs the following steps:
    1. Detect candidate cells using LoG filtering (`macro_track2`).
    2. If no cells are found, return empty outputs.
    3. Classify morphology and TME state for each detected cell (`classify_tumor_cells`).
    4. Track cells over time using a Kalman filter (`track_tumor_cells2`).
    5. Connect fragmented tracklets (`connect_tracklets`).
    6. Assemble per-track properties and embeddings into DataFrames
       (`build_track_dataframes`).

    @ Author: André Lasses Armatowski, Madeleine Skeppås, Sven Nelander

    """

    sigmah = 16.0
    cutoff = 2e-4

    scaler1 = 1 + detection_sensitivity * 9  # 10 - 1
    scaler2 = 1 + detection_sensitivity * 29  # 30 - 1

    cellsx, cellsy, intensity, feat, vascc, all_empty = macro_track2(
        gbm,
        vasc,
        sigmah / scaler1,
        cutoff * scaler2,
        blocksize,
        mode="normal",
        debug=False,
    )

    if all_empty:
        return None

    properties = classify_tumor_cells(
        feat, vascc, blocksize, net_const, tme_net_const, stack_index
    )

    (
        traX,
        traY,
        x_hat_history,
        phenotypes,
        phenonames,
        Kn_history,
        startidx,
    ) = track_tumor_cells2(cellsx, cellsy, properties)

    traX, traY, phenotypes = connect_tracklets(traX, traY, phenotypes)

    tracks_with_props = build_track_dataframes(
        traX, traY, phenotypes, phenonames, properties, startidx
    )

    return tracks_with_props
