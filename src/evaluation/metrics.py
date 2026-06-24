# ==============================================================================
# Evaluation Metrics
# ==============================================================================
# Extracted and improved from 01k / 01l notebooks.
#
# Key improvements over the notebook code:
#   - MMD uses the **median heuristic** to automatically select gamma instead
#     of iterating over a fixed list of [0.1, 0.5, 1.0, 2.0].
#   - physics_filter accepts feature indices as arguments — no hardcoding.
#   - All functions are pure NumPy/sklearn; no plotting, no side-effects.
#
# References:
#   Gretton et al. (2012) "A Kernel Two-Sample Test". JMLR.
#   Median heuristic: Schölkopf (1998), Garreau et al. (2017).
# ==============================================================================

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Median heuristic
# ------------------------------------------------------------------------------

def median_heuristic_gamma(X: np.ndarray, subsample: int = 2000) -> float:
    """
    Computes the RBF kernel bandwidth gamma using the **median heuristic**.

    gamma = 1 / (2 * median(||x_i - x_j||²))

    The median of pairwise squared Euclidean distances in X is used as the
    bandwidth, providing a data-driven alternative to fixed gamma grids.

    Args:
        X:          Data matrix of shape (N, D). Real latent vectors are typical.
        subsample:  Maximum number of rows to use when computing pairwise
                    distances (keeps memory bounded for large N). Default 2000.

    Returns:
        Scalar gamma value suitable for ``sklearn.metrics.pairwise.rbf_kernel``.
    """
    if len(X) > subsample:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(X), size=subsample, replace=False)
        X = X[idx]

    # Pairwise squared Euclidean distances: cdist returns L2, so we square it
    pairwise_sq_dists = cdist(X, X, metric="sqeuclidean")

    # Take upper triangle (exclude zero diagonal) to get unique pairs
    upper = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]

    median_sq = np.median(upper)
    if median_sq < 1e-10:
        log.warning(
            "Median pairwise distance is near zero. Falling back to gamma=1.0."
        )
        return 1.0

    gamma = 1.0 / (2.0 * median_sq)
    log.info(f"[MMD] Median heuristic → median_sq_dist={median_sq:.6f}, gamma={gamma:.6f}")
    return float(gamma)


# ------------------------------------------------------------------------------
# MMD²
# ------------------------------------------------------------------------------

def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> Tuple[float, float]:
    """
    Computes the Maximum Mean Discrepancy squared (MMD²) between two sets of
    vectors using an RBF (Gaussian) kernel.

    If ``gamma`` is ``None``, the median heuristic is applied to ``X`` to
    compute an appropriate bandwidth automatically.

    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    Args:
        X:     Real samples, shape (N, D). Typically real encoder latent vectors.
        Y:     Synthetic samples, shape (M, D).
        gamma: RBF kernel gamma. If ``None`` the median heuristic is used.

    Returns:
        Tuple of (mmd_squared, gamma_used).
    """
    if gamma is None:
        gamma = median_heuristic_gamma(X)

    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = float(np.mean(XX) + np.mean(YY) - 2.0 * np.mean(XY))
    return mmd2, gamma


def evaluate_mmd(
    real_latents: np.ndarray,
    fake_latents: np.ndarray,
    n_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluates MMD² between real and synthetic latent vectors, reporting both
    the median-heuristic gamma and the resulting MMD² value.

    Args:
        real_latents: Real encoder latent vectors, shape (N, latent_dim).
        fake_latents: Synthetic generator output, shape (M, latent_dim).
        n_samples:    Maximum number of samples to use (for efficiency).

    Returns:
        Dict with keys ``"mmd2"`` and ``"gamma"``.
    """
    n = min(n_samples, len(real_latents), len(fake_latents))

    rng = np.random.default_rng(seed=42)
    real_sub = real_latents[rng.choice(len(real_latents), n, replace=False)]
    fake_sub = fake_latents[rng.choice(len(fake_latents), n, replace=False)]

    mmd2, gamma = compute_mmd(real_sub, fake_sub, gamma=None)

    log.info(f"[MMD] γ={gamma:.6f}  MMD²={mmd2:.6f}  (n={n})")
    return {"mmd2": mmd2, "gamma": gamma}


# ------------------------------------------------------------------------------
# Physics rejection filter
# ------------------------------------------------------------------------------

def physics_filter(
    sequences: np.ndarray,
    conditions: np.ndarray,
    soc_idx: int,
    volt_idx: int,
    max_soc_drops: int = 2,
    min_soc_variance: float = 0.0005,
    max_volt_drop: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hard rejection sampling: removes synthetic sequences that violate
    the three physics rules used across all notebooks.

    Rules:
        1. SOC must not decrease by more than ``max_soc_drops`` ticks of > 0.01.
        2. SOC curve must not be essentially flat (variance > ``min_soc_variance``).
        3. End voltage must not fall more than ``max_volt_drop`` below start voltage.

    Args:
        sequences:        Synthetic sequences (N, seq_len, num_features).
        conditions:       Conditioning capacity array (N, 1).
        soc_idx:          Column index of SOC feature.
        volt_idx:         Column index of Average_Cell_Voltage feature.
        max_soc_drops:    Maximum number of allowable SOC decreases > 0.01.
        min_soc_variance: Minimum acceptable SOC variance.
        max_volt_drop:    Maximum acceptable end-start voltage drop.

    Returns:
        Tuple of (filtered_sequences, filtered_conditions).
    """
    def _passes(seq: np.ndarray) -> bool:
        soc  = seq[:, soc_idx]
        volt = seq[:, volt_idx]
        if np.sum(np.diff(soc) < -0.01) > max_soc_drops:
            return False
        if np.var(soc) < min_soc_variance:
            return False
        if volt[-1] < volt[0] - max_volt_drop:
            return False
        return True

    mask = np.array([_passes(s) for s in sequences])
    n_pass = int(mask.sum())
    log.info(
        f"[PhysicsFilter] Passed {n_pass}/{len(sequences)} "
        f"({100*n_pass/max(len(sequences),1):.1f}%) synthetic samples."
    )
    return sequences[mask], conditions[mask]
