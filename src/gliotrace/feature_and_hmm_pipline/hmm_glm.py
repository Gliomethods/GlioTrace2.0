import copy
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax
from tqdm import tqdm


# --------- CNN-HMM -------
#
# ---------------------------------------------------
# ----------------- E - step ------------------------
# ---------------------------------------------------
# The E-step is done for each trajectory separately. It is connected
# across different trajectories by using the same transition probability
# parameters (GLM models or a global A). The goal is to compute/update
# the probabilities needed to take the expectation and is trajectory specific:
#
# gamma[t, j] = Pr(Z_t = j | Y_(1:T))                  (T, K)
# xi[t, i, j] = Pr(Z_t+1 = j, Z_t = i | Y_(1:T))       (T-1, K, K)
#
# We use the forward/backward algorithm where 'alpha' carries past info
# and 'beta' carries future info.
#
# Dimensions:
# T: length of a trajectory
# K: number of hidden states (e.g., 6)
# F: dimension of feature data for the GLM transitions
#
# Variables:
# cnn_out: observed CNN outputs for one trajectory (logits) (T, K)
# trajectory: GLM covariates for transitions                (T, F)
#
# Important NOTE: Work in log-probabilities with small epsilons to avoid
# underflow, log(0), and division by zero.


# ------------- Helper functions ------------------

def calc_A_global(xis, n_states):
    """
    Calculate a global transition matrix using the posterior hmm informed information
    xis: Dict[key] -> (T-1, K, K)
    n_states = K

    Author: Madeleine Skeppås
    """
    A_num = np.zeros((n_states, n_states))

    for _, xi in xis.items():
        A_num += np.sum(xi, axis=0)

    A_row_sums = np.sum(A_num, axis=1, keepdims=True)
    A = A_num / A_row_sums

    return A


def evaluate_models_on_trajectories(models, trajectory, K, eps=1e-6, A=None):
    """
    Evaluate transition probabilities in log-space.

    models:     list[K] of sklearn LogisticRegression or None
    trajectory: (T, F)
    A:          optional global transition matrix (K, K)

    Returns: (T, K, K) array of log probs.

    Author: André Lasses Armatowski
    """
    T, F = trajectory.shape
    A_log = np.zeros((T, K, K))

    if models is None and A is None:
        stay = 0.9
        move = (1.0 - stay) / (K - 1)

        # Build base transition matrix (K, K)
        A_base = np.full((K, K), move)
        np.fill_diagonal(A_base, stay)

        # Convert to log-space
        A_log_base = np.log(A_base + eps)

        # Repeat across all timesteps (T, K, K)
        A_log = np.broadcast_to(A_log_base, (T, K, K)).copy()
        return A_log

    elif models is None and A is not None:
        A_log = np.log(A + eps)
        A_log_broad = np.broadcast_to(A_log, (T, K, K)).copy()
        return A_log_broad

    else:
        for i, model in enumerate(models):
            # trajectory: (T, F), coef_: (K, F), intercept_: (K,)
            logits = trajectory @ model.coef_.T + model.intercept_[None, :]
            A_log[:, i, :] = log_softmax(logits, axis=1)

        return A_log


# -----------------------------------------------------------
# ------------------ Forward / backward ---------------------
# -----------------------------------------------------------

def forward_backward(
    cnn_out,
    trajectory,
    glm_models,
    pi,
    eps=1e-12,
    A=None,
):
    """
    Run forward-backward to compute gamma and xi under CNN emissions.

    cnn_out:      (T, K) logits from the network
    trajectory:   (T, F) transition covariates
    glm_models:   list[K] or None
    pi:           (K,)
    A:            optional global transition matrix (K, K)

    Returns:
        gamma: (T, K)
        xi:    (T-1, K, K)
        log_lik: float

    Author: André Lasses Armatowski
    """
    T = trajectory.shape[0]
    K = int(np.asarray(pi).shape[0])

    # Initialize parameters
    log_alpha = np.zeros((T, K))
    log_beta = np.zeros((T, K))
    log_xi = np.zeros((T - 1, K, K))

    # ---- Emissions from CNN ----
    x = np.asarray(cnn_out, dtype=float)

    # log p(y_t | z_t=k) from logits
    log_em = log_softmax(x, axis=1)

    # ---- Transitions: log p(z_{t+1}=j | z_t=i, features_t) ----
    log_A = evaluate_models_on_trajectories(glm_models, trajectory, K, eps, A)

    # ---- Initial distribution ----
    log_pi = np.log(np.maximum(np.asarray(pi, dtype=float), eps))
    log_pi -= np.logaddexp.reduce(log_pi)

    # Alpha recursion
    log_alpha[0] = log_pi + log_em[0]
    for t in range(1, T):
        log_alpha[t] = log_em[t] + np.logaddexp.reduce(
            log_alpha[t - 1][:, None] + log_A[t - 1], axis=0
        )

    # Log-likelihood
    log_lik = float(np.logaddexp.reduce(log_alpha[T - 1]))

    # Beta recursion
    log_beta[T - 1, :] = 0.0
    for t in reversed(range(T - 1)):
        log_beta[t] = np.logaddexp.reduce(
            log_A[t] + log_em[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1
        )

    # Posterior gamma
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # Posterior xi
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None]
            + log_A[t]
            + log_em[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        log_xi_t -= np.logaddexp.reduce(log_xi_t.ravel())
        log_xi[t] = log_xi_t

    xi = np.exp(log_xi)

    return gamma, xi, log_lik


# ---------------------------------------------------
# ----------------- M - step ------------------------
# ---------------------------------------------------
# The M-step maximizes the (expected) complete-data log-likelihood.
# Here: re-fit GLM transitions.


def update_transitions(trajectories, xis, glm_iters=500, prev_models=None):
    """
    trajectories: shape (dict[S], T, F)
    xis: shape (dict[S], T, K, K)
    Returns:
        List of K multilogistic models

    Author: André Lasses Armatowski
    """

    # Extract any one item (first trajectory)
    first_xi = next(iter(xis.values()))

    K = first_xi.shape[1]
    models = []

    # One set of coefficients for each "from" state
    for i in range(K):

        # Create training data
        X, y, sample_weights = [], [], []
        for cell_id, xi in xis.items():
            feats = trajectories[cell_id]
            T = xi.shape[0]

            for t in range(T - 1):
                for j in range(K):
                    X.append(feats[t])
                    y.append(j)
                    sample_weights.append(xi[t, i, j])

        if prev_models is not None and len(prev_models) > i:
            clf = prev_models[i]
        else:
            clf = LogisticRegression()

        clf.set_params(
            solver="lbfgs",
            warm_start=True,
            C=1e12,
            tol=0.001,
            max_iter=glm_iters,
        )

        clf.fit(X, y, sample_weight=sample_weights)
        models.append(clf)

    return models

# -------------------------------------------------------
# ------------------- Format output ----------------------
# -------------------------------------------------------


def _key_to_cols(key):
    # key is expected to be (exp, roi, cellID)
    if isinstance(key, tuple) and len(key) == 3:
        exp, roi, cellID = key
        return {"exp": exp, "roi": roi, "cellID": cellID}
    return {"key": str(key)}


def _gammas_to_long_df(gammas, state_names):
    frames = []
    for key, g in gammas.items():
        g = np.asarray(g)  # (T, K)
        df = pd.DataFrame(g, columns=state_names)
        df.insert(0, "t", np.arange(df.shape[0], dtype=int))

        key_cols = _key_to_cols(key)
        # Insert exp/roi/cellID at front (preserve order exp, roi, cellID if present)
        # _key_to_cols returns dict in that order
        for col_name, col_val in reversed(list(key_cols.items())):
            df.insert(0, col_name, col_val)

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hmm_glm(
    trajectories,
    cnn_outputs,
    pi,
    K=6,
    max_iter=10,
    glm_iters=500,
    eps_conv=1e-6,
    glm_models=None,
    A=None,
    state_names=None,
    patience=10,            # <-- stop if no improvement for this many EM steps
):
    """
    CNN-HMM with feature-conditioned transitions (GLM) and fixed CNN emissions.

    trajectories: dict[key] -> (T, F)
    cnn_outputs:  dict[key] -> (T, K) logits 
    pi:           (K,)
    glm_models:   list[K] or None
    A:            optional global transition matrix (K, K)

    Returns best (highest full likelihood) parameters encountered.

    @ Author: André Lasses Armatowski
    """

    best_lik = -np.inf
    no_improve = 0

    # Store best snapshot (deep copies for mutable objects)
    best_pi = None
    best_glm_models = None
    best_A_global = None
    best_gammas = None

    A_global = None

    if state_names is None:
        state_names = [f"state_{k+1}" for k in range(K)]

    for it in tqdm(range(max_iter), desc="Performing CNN-HMM - GLM algorithm", unit="EM-step"):

        gammas = {}
        xis = {}
        full_lik = 0.0

        # ---- E-step ----
        for key in trajectories:
            cnn_out = cnn_outputs[key]
            F = trajectories[key]

            gamma, xi, log_lik = forward_backward(
                cnn_out=cnn_out,
                trajectory=F,
                glm_models=glm_models,
                pi=pi,
                eps=1e-12,
                A=A
            )
            gammas[key] = gamma
            xis[key] = xi
            full_lik += float(log_lik)

        A_global = calc_A_global(xis, K)

        # ---- track best ----
        if full_lik > best_lik + eps_conv:
            best_lik = full_lik
            no_improve = 0

            best_pi = np.array(pi, copy=True)
            best_glm_models = copy.deepcopy(glm_models)
            best_A_global = np.array(A_global, copy=True)
            best_gammas = copy.deepcopy(gammas)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"Early stop: no improvement for {patience} steps. Returning best model.")
            break

        # ---- M-step ----
        # Update pi from responsibilities at t=0
        pi_stack = np.vstack([g[0] for g in gammas.values()]).astype(float)
        pi = pi_stack.mean(axis=0)
        pi = np.maximum(pi, 1e-12)
        pi /= pi.sum()

        # Update feature-conditioned transitions
        glm_models = update_transitions(
            trajectories,
            xis,
            prev_models=glm_models,
            glm_iters=glm_iters,
        )

    # ---- if never improved, fall back to final ----
    if best_pi is None:
        best_pi = np.array(pi, copy=True)
        best_glm_models = copy.deepcopy(glm_models)
        best_A_global = np.array(
            A_global, copy=True) if A_global is not None else None
        best_gammas = copy.deepcopy(gammas)

    # ----------- FORMAT OUTPUTS (ALWAYS) using BEST snapshot -----------
    pi_ser = pd.Series(np.asarray(best_pi).squeeze(),
                       index=state_names, name="pi")

    A_df = pd.DataFrame(np.asarray(best_A_global),
                        index=state_names, columns=state_names)

    gammas_long = _gammas_to_long_df(best_gammas, state_names)

    return pi_ser, best_glm_models, A_df, gammas_long
