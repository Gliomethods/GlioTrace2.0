import pandas as pd
from gliotrace.feature_and_hmm_pipline.hmm_shrunken import evaluate_models_on_trajectories
import numpy as np


def viterbi_from_logs(pi, A_log, B_log, eps=1e-12):
    T, K = B_log.shape
    logpi = np.log(np.clip(pi, eps, 1.0))

    delta = np.empty((T, K), dtype=float)
    psi = np.empty((T, K), dtype=np.int64)

    delta[0] = logpi + B_log[0]
    psi[0] = -1

    for t in range(1, T):
        scores = delta[t-1][:, None] + A_log[t-1]   # (K,K)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = B_log[t] + scores[psi[t], np.arange(K)]

    path = np.empty(T, dtype=np.int64)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-1, 0, -1):
        path[t-1] = psi[t, path[t]]
    return path


def viterbi_paths_all_tracks(trajectories, cnn_outputs_log, pi, glm_models, K, state_names=None, eps=1e-12, A=None):
    """
    trajectories: dict[key] -> (T,F)
    cnn_outputs_log: dict[key] -> (T,K) log probs (your out)
    returns: DataFrame with columns [key, t, state, state_name]
    """
    if state_names is None:
        state_names = [f"state_{k}" for k in range(K)]

    rows = []
    for key in trajectories.keys():
        X = trajectories[key]          # (T,F)
        B_log = cnn_outputs_log[key]   # (T,K) log probs

        # transitions in log-space (T,K,K) using your existing function
        A_log = evaluate_models_on_trajectories(glm_models, X, K, eps=eps, A=A)

        path = viterbi_from_logs(pi, A_log, B_log, eps=eps)  # (T,)
        rows.append(pd.DataFrame({
            "exp": key[0],
            "roi": key[1],
            "cellID": key[2],
            "t": np.arange(len(path)),
            "state": path,
            "state_name": [state_names[s] for s in path],
        }))

    return pd.concat(rows, ignore_index=True)


def map_viterbi_t_to_time_and_merge(data_feat, viterbi_df, K=None):
    df = data_feat.copy()

    # Ensure types align
    for c in ["exp", "roi", "cellID"]:
        df[c] = df[c].astype(int)
        viterbi_df[c] = viterbi_df[c].astype(int)

    df["time"] = df["time"].astype(int)
    viterbi_df["t"] = viterbi_df["t"].astype(int)

    # Build per-track "rank within track by increasing time": 0,1,2,...
    df = df.sort_values(["exp", "roi", "cellID", "time"]).copy()
    df["_t"] = df.groupby(["exp", "roi", "cellID"]).cumcount()

    # Merge viterbi state onto this rank
    v = viterbi_df.rename(columns={"t": "_t", "state": "viterbi_state"}).copy()

    out = df.merge(v[["exp", "roi", "cellID", "_t", "viterbi_state"]],
                   on=["exp", "roi", "cellID", "_t"], how="left")

    # cleanup
    out["viterbi_state"] += 1
    out = out.drop(columns=["_t"])
    return out
