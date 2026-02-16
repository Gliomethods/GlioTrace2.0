from gliotrace.feature_and_hmm_pipline.feature_construction import feature_construction
from gliotrace.feature_and_hmm_pipline.hmm_shrunken import HMM_glm_cnn
from gliotrace.feature_and_hmm_pipline.format_data import format_data
from gliotrace.feature_and_hmm_pipline.clean_tracks import filter_features_3
from gliotrace.feature_and_hmm_pipline.viterbi_paths import viterbi_paths_all_tracks, map_viterbi_t_to_time_and_merge

from gliotrace.initalize_class.defaults import HARD_CODED_FEATURES, NON_SCALE_COLUMNS, SOFTMAX_COLUMNS

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def _init_sticky_A(K, stay=0.95):
    "Intial A, can be changed"
    move = (1.0 - stay) / (K - 1)
    A = np.full((K, K), move, dtype=float)
    np.fill_diagonal(A, stay)
    return A


def hmm_pipeline(data, fcols, hmm_param):
    """
    Prepares features and runs HMM model with given features and logits from the CNN

    Author: Linnea Hallin, André Lasses Armatowski (adapted)
    """

    if fcols is None or len(fcols) == 0:
        raise ValueError("fcols must be a non-empty list of feature columns")

    # ----------- Create features -----------
    print("--- Creating features ---")
    frames = []
    for exp in data["exp"].unique():
        exp_data = data.loc[data["exp"] == exp].copy()
        features_exp = feature_construction(exp_data)
        frames.append(features_exp)

    data_feat = pd.concat(frames, ignore_index=True)

    data_feat_unfiltered = data_feat.copy()

    # ----------- Filter tracks on non-NA and size -----------
    data_feat = filter_features_3(
        data_feat,
        fcols=fcols,
        hard_coded_features=HARD_CODED_FEATURES,
        min_timepoints=2,
        verbose=True,
    )
    if data_feat is None:
        raise RuntimeError("No usable tracks after filtering")

    # ----------- Define number of states -----------
    softmax_columns = SOFTMAX_COLUMNS
    n_states = len(softmax_columns)

    # ----------- Create initial A -----------
    # NOTE: Can be changed
    print("--- Intialize transitions ---")
    A = _init_sticky_A(n_states, stay=0.95)

    # ----------- Scale and format -----------
    print("--- Final Preperations for GLM-HMM ---")

    exclude = set(NON_SCALE_COLUMNS) | set(HARD_CODED_FEATURES)
    scale_cols = [c for c in fcols if c not in exclude]

    if len(scale_cols) > 0:
        data_feat.loc[:, scale_cols] = StandardScaler(
        ).fit_transform(data_feat[scale_cols])

    # Put data into correct format for HMM (trajectories-only version)
    track_data, cnn_outputs = format_data(
        trajectories=data_feat,
        columns=fcols,
        softmax_cols=softmax_columns,
    )

    # ----------- hyperparameters for glm ----------
    max_iter = hmm_param["em_iter"]
    penalty = hmm_param["penalty"]
    glm_iter = hmm_param["glm_iter"]
    eps = hmm_param["eps"]

    # ----------- Initialize pi -----------
    # NOTE: Can be changed
    pi0 = np.ones(n_states, dtype=float) / float(n_states)

    # ----------- Fit HMM -----------
    print("--- Running GLM-HMM ---")
    pi, glm_models, A_global, gammas, lik = HMM_glm_cnn(
        track_data,
        cnn_outputs,
        pi=pi0,
        max_iter=max_iter,
        glm_iters=glm_iter,
        penalty=penalty,
        eps_conv=eps,
        A=A,
        state_names=softmax_columns,
        patience=10
    )

    print("--- Computing Viterbi Paths ---")

    viterbi_df = viterbi_paths_all_tracks(
        trajectories=track_data,
        cnn_outputs_log=cnn_outputs,
        pi=np.asarray(pi),
        glm_models=glm_models,
        K=n_states
    )

    data_feat = map_viterbi_t_to_time_and_merge(
        data_feat, viterbi_df, K=n_states)

    return data_feat_unfiltered, data_feat, pi, glm_models, A_global, gammas, lik
