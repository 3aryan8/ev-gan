# ==============================================================================
# Evaluation Metrics
# ==============================================================================
# Extracted and improved from 01k / 01l notebooks.
#
# Key improvements over the notebook code:
#   - MMD uses the **median heuristic** for bandwidth selection.
#   - physics_filter accepts feature indices — no hardcoding.
#   - Wasserstein-1 distance (per feature marginal).
#   - Jensen-Shannon divergence (per feature marginal, histogram-based).
#   - Autocorrelation fidelity (mean-absolute-difference of ACF).
#   - save_metrics_json — persists any metric dict to JSON with timestamp.
#   - compute_all_metrics — single orchestrator for the full metric set.
#
# Data-leakage policy
# -------------------
#   Metrics comparing real vs. synthetic MUST receive held-out test-set arrays
#   (X_test, test_latents). The one exception is evaluate_mmd(train_latents)
#   kept for backward-compat and clearly named. compute_all_metrics() computes
#   MMD on both splits, labelled "train_split" and "test_split" respectively.
#
# References:
#   Gretton et al. (2012) "A Kernel Two-Sample Test". JMLR.
#   Median heuristic: Schölkopf (1998), Garreau et al. (2017).
#   Wasserstein-1: Villani (2008) "Optimal Transport".
#   JS divergence: Lin (1991) "Divergence measures based on the Shannon entropy".
# ==============================================================================

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Median heuristic
# ------------------------------------------------------------------------------

def median_heuristic_gamma(X: np.ndarray, subsample: int = 2000) -> float:
    """
    Computes the RBF kernel bandwidth gamma using the **median heuristic**.

    gamma = 1 / (2 * median(||x_i - x_j||²))

    Args:
        X:          Data matrix of shape (N, D).
        subsample:  Max rows for pairwise distances (memory bound). Default 2000.

    Returns:
        Scalar gamma for ``sklearn.metrics.pairwise.rbf_kernel``.
    """
    if len(X) > subsample:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(X), size=subsample, replace=False)
        X = X[idx]

    pairwise_sq_dists = cdist(X, X, metric="sqeuclidean")
    upper = pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)]
    median_sq = np.median(upper)

    if median_sq < 1e-10:
        log.warning("Median pairwise distance is near zero. Falling back to gamma=1.0.")
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
    Computes MMD² between two sample sets using an RBF (Gaussian) kernel.

    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    Args:
        X:     Real samples (N, D).
        Y:     Synthetic samples (M, D).
        gamma: RBF gamma. If ``None`` the median heuristic is used.

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
    Evaluates MMD² between real and synthetic latent vectors.

    Args:
        real_latents: Real encoder latent vectors (N, latent_dim).
        fake_latents: Synthetic generator output (M, latent_dim).
        n_samples:    Maximum samples to use. Default 1000.

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
    the three physics rules.

    Rules:
        1. SOC must not decrease by more than ``max_soc_drops`` ticks of > 0.01.
        2. SOC curve must not be essentially flat (variance > ``min_soc_variance``).
        3. End voltage must not fall more than ``max_volt_drop`` below start voltage.

    Args:
        sequences:        Synthetic sequences (N, seq_len, num_features).
        conditions:       Conditioning capacity array (N, 1).
        soc_idx:          Column index of SOC feature.
        volt_idx:         Column index of Average_Cell_Voltage feature.
        max_soc_drops:    Max allowable SOC decreases > 0.01.
        min_soc_variance: Min acceptable SOC variance.
        max_volt_drop:    Max acceptable end-start voltage drop.

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


def physics_pass_rate(
    sequences: np.ndarray,
    soc_idx: int,
    volt_idx: int,
    max_soc_drops: int = 2,
    min_soc_variance: float = 0.0005,
    max_volt_drop: float = 0.01,
) -> Dict[str, float]:
    """
    Computes the per-rule and combined pass-rate for the physics filter,
    returning only statistics (no filtered arrays).

    Args:
        sequences:        Synthetic sequences (N, seq_len, num_features).
        soc_idx:          Column index of SOC feature.
        volt_idx:         Column index of Average_Cell_Voltage feature.
        max_soc_drops:    Max allowable SOC drops > 0.01. Default 2.
        min_soc_variance: Min acceptable SOC variance. Default 0.0005.
        max_volt_drop:    Max acceptable end-start voltage drop. Default 0.01.

    Returns:
        Dict with keys:
            ``"rule_soc_monotonicity"``  — fraction passing rule 1,
            ``"rule_soc_variance"``      — fraction passing rule 2,
            ``"rule_volt_stability"``    — fraction passing rule 3,
            ``"combined"``               — fraction passing all three,
            ``"n_total"``                — total sequences evaluated.
    """
    n = len(sequences)
    soc_all  = sequences[:, :, soc_idx]
    volt_all = sequences[:, :, volt_idx]

    r1 = np.array([np.sum(np.diff(s) < -0.01) <= max_soc_drops for s in soc_all])
    r2 = np.var(soc_all, axis=1) >= min_soc_variance
    r3 = (volt_all[:, -1] - volt_all[:, 0]) >= -max_volt_drop
    combined = r1 & r2 & r3

    rates = {
        "rule_soc_monotonicity": float(r1.mean()),
        "rule_soc_variance":     float(r2.mean()),
        "rule_volt_stability":   float(r3.mean()),
        "combined":              float(combined.mean()),
        "n_total":               float(n),
    }
    log.info(
        f"[PhysicsPassRate] combined={rates['combined']:.3f}  "
        f"soc_mono={rates['rule_soc_monotonicity']:.3f}  "
        f"soc_var={rates['rule_soc_variance']:.3f}  "
        f"volt={rates['rule_volt_stability']:.3f}"
    )
    return rates


# ------------------------------------------------------------------------------
# Wasserstein-1 distance (per feature marginal)
# ------------------------------------------------------------------------------

def compute_wasserstein_per_feature(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    n_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Computes the 1-D Wasserstein-1 (Earth Mover's) distance between the
    marginal distributions of real and synthetic sequences per feature.

    Sequences are flattened across the time axis so we compare the full
    empirical distribution of each feature value.

    No data leakage: call with held-out test-set sequences only.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_indices: Mapping feature_name → column index.
        n_samples:       Max samples per feature (``None`` = all). Default 5000.
        random_state:    RNG seed. Default 42.

    Returns:
        Dict mapping feature_name → Wasserstein-1 distance (float ≥ 0).
    """
    rng = np.random.default_rng(seed=random_state)
    results: Dict[str, float] = {}

    for name, idx in feature_indices.items():
        real_vals = real_sequences[:, :, idx].flatten()
        fake_vals = fake_sequences[:, :, idx].flatten()

        if n_samples is not None:
            n_r = min(n_samples, len(real_vals))
            n_f = min(n_samples, len(fake_vals))
            real_vals = rng.choice(real_vals, size=n_r, replace=False)
            fake_vals = rng.choice(fake_vals, size=n_f, replace=False)

        w1 = float(wasserstein_distance(real_vals, fake_vals))
        results[name] = w1
        log.info(f"[Wasserstein] {name}: W1={w1:.6f}")

    return results


# ------------------------------------------------------------------------------
# Jensen-Shannon divergence (per feature marginal, histogram-based)
# ------------------------------------------------------------------------------

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p‖q) with epsilon zero-avoidance. Both arrays must sum to 1."""
    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def compute_js_divergence_per_feature(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    n_bins: int = 50,
    n_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Computes histogram-based Jensen-Shannon (JS) divergence between real and
    synthetic marginal distributions per feature.

    JS is symmetric, bounded in [0, log(2)], and robust to support mismatch.
    Both histograms share bin edges derived from the union of both value ranges.

    JS(P‖Q) = 0.5·KL(P‖M) + 0.5·KL(Q‖M),  M = 0.5·(P+Q)

    No data leakage: call with held-out test-set sequences.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_indices: Mapping feature_name → column index.
        n_bins:          Number of histogram bins. Default 50.
        n_samples:       Max samples per feature (``None`` = all). Default 5000.
        random_state:    RNG seed. Default 42.

    Returns:
        Dict mapping feature_name → JS divergence ∈ [0, log(2)].
    """
    rng = np.random.default_rng(seed=random_state)
    results: Dict[str, float] = {}

    for name, idx in feature_indices.items():
        real_vals = real_sequences[:, :, idx].flatten()
        fake_vals = fake_sequences[:, :, idx].flatten()

        if n_samples is not None:
            n_r = min(n_samples, len(real_vals))
            n_f = min(n_samples, len(fake_vals))
            real_vals = rng.choice(real_vals, size=n_r, replace=False)
            fake_vals = rng.choice(fake_vals, size=n_f, replace=False)

        global_min = min(real_vals.min(), fake_vals.min())
        global_max = max(real_vals.max(), fake_vals.max())
        if global_max - global_min < 1e-10:
            log.warning(f"[JS] Feature '{name}' has near-zero range — JS set to 0.")
            results[name] = 0.0
            continue

        bins = np.linspace(global_min, global_max, n_bins + 1)
        p, _ = np.histogram(real_vals, bins=bins, density=False)
        q, _ = np.histogram(fake_vals, bins=bins, density=False)
        p = p.astype(float) / p.sum()
        q = q.astype(float) / q.sum()

        m = 0.5 * (p + q)
        js = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
        results[name] = float(js)
        log.info(f"[JS] {name}: JS={js:.6f}  (n_bins={n_bins})")

    return results


# ------------------------------------------------------------------------------
# Autocorrelation fidelity
# ------------------------------------------------------------------------------

def _mean_acf(sequences: np.ndarray, feature_idx: int, max_lag: int) -> np.ndarray:
    """Mean sample ACF across all sequences for one feature at lags 1..max_lag."""
    n_seqs = sequences.shape[0]
    acf_matrix = np.zeros((n_seqs, max_lag))

    for i, seq in enumerate(sequences):
        x = seq[:, feature_idx].astype(float)
        x -= x.mean()
        var = np.var(x)
        if var < 1e-10:
            acf_matrix[i, :] = 0.0
            continue
        for lag in range(1, max_lag + 1):
            acf_matrix[i, lag - 1] = float(np.mean(x[lag:] * x[:-lag]) / var)

    return acf_matrix.mean(axis=0)


def compute_autocorr_fidelity(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    max_lag: int = 20,
    n_samples: Optional[int] = 1000,
    random_state: int = 42,
) -> Dict:
    """
    Mean absolute difference (MAD) between real and synthetic mean ACF curves,
    averaged over lags 1..``max_lag`` per feature.

    A value near 0 indicates the synthetic sequences reproduce the temporal
    correlation structure of the real data.

    No data leakage: call with held-out test-set sequences.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_indices: Mapping feature_name → column index.
        max_lag:         Maximum lag. Default 20.
        n_samples:       Subsample cap (``None`` = all). Default 1000.
        random_state:    RNG seed. Default 42.

    Returns:
        Dict mapping feature_name → MAD float, plus key ``"_acf_curves"``
        containing nested dicts of ``{"real": [...], "fake": [...]}`` per feature
        for use by visualization functions.
    """
    rng = np.random.default_rng(seed=random_state)
    n_r = min(n_samples, len(real_sequences)) if n_samples else len(real_sequences)
    n_f = min(n_samples, len(fake_sequences)) if n_samples else len(fake_sequences)
    real_sub = real_sequences[rng.choice(len(real_sequences), n_r, replace=False)]
    fake_sub = fake_sequences[rng.choice(len(fake_sequences), n_f, replace=False)]

    results: Dict = {}
    acf_curves: Dict[str, Dict[str, List[float]]] = {}

    for name, idx in feature_indices.items():
        acf_real = _mean_acf(real_sub, idx, max_lag)
        acf_fake = _mean_acf(fake_sub, idx, max_lag)
        mad = float(np.mean(np.abs(acf_real - acf_fake)))
        results[name] = mad
        acf_curves[name] = {"real": acf_real.tolist(), "fake": acf_fake.tolist()}
        log.info(f"[AutocorrFidelity] {name}: MAD={mad:.6f}  (max_lag={max_lag})")

    results["_acf_curves"] = acf_curves
    return results


# ------------------------------------------------------------------------------
# Metrics JSON persistence
# ------------------------------------------------------------------------------

def save_metrics_json(
    metrics: Dict,
    save_path: str,
    run_id: Optional[str] = None,
) -> None:
    """
    Serialises a flat or nested metrics dict to JSON at ``save_path``,
    adding an ISO-8601 UTC timestamp and optional run identifier.

    NumPy scalars/arrays are coerced to Python-native types automatically.

    Args:
        metrics:   Dict of metric name → value (float, int, str, list, or
                   nested dicts). NumPy arrays must be converted to lists first.
        save_path: Path for the output ``*.json`` file.
        run_id:    Optional experiment identifier added under key ``"run_id"``.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    def _coerce(obj):
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_coerce(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        **(_coerce({"run_id": run_id}) if run_id else {}),
        "metrics": _coerce(metrics),
    }

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)

    log.info(f"[Metrics] Saved → {save_path}")


# ------------------------------------------------------------------------------
# Orchestrator: compute all metrics
# ------------------------------------------------------------------------------

def compute_all_metrics(
    real_sequences_test: np.ndarray,
    fake_sequences: np.ndarray,
    real_latents_train: np.ndarray,
    real_latents_test: np.ndarray,
    fake_latents: np.ndarray,
    feature_indices: Dict[str, int],
    soc_idx: int,
    volt_idx: int,
    n_bins: int = 50,
    autocorr_max_lag: int = 20,
    mmd_n_samples: int = 1000,
    random_state: int = 42,
) -> Dict:
    """
    Computes the full PIC-GAN evaluation metric suite:

    1. MMD² on train latents (backward-compat; labelled ``"train_split"``).
    2. MMD² on test latents (leakage-free; labelled ``"test_split"``).
    3. Wasserstein-1 per feature (test sequences).
    4. JS divergence per feature (test sequences).
    5. Autocorrelation fidelity MAD per feature (test sequences).
    6. Physics pass-rate — per-rule and combined (synthetic sequences).

    No data leakage: metrics 2–6 use only held-out test-set arrays.

    Args:
        real_sequences_test: Real test sequences (N_test, seq_len, num_features).
        fake_sequences:      Synthetic sequences (M, seq_len, num_features).
        real_latents_train:  Train-split encoder latents (N_train, D).
        real_latents_test:   Test-split encoder latents (N_test, D).
        fake_latents:        Synthetic generator latents (M, D).
        feature_indices:     Dict mapping feature name → column index.
        soc_idx:             Column index of the SOC feature.
        volt_idx:            Column index of the voltage feature.
        n_bins:              Histogram bins for JS divergence. Default 50.
        autocorr_max_lag:    Max lag for ACF computation. Default 20.
        mmd_n_samples:       Subsample cap for MMD. Default 1000.
        random_state:        Reproducibility seed. Default 42.

    Returns:
        Nested dict suitable for passing directly to ``save_metrics_json``.
    """
    log.info("[EvalSuite] Computing full metric suite …")

    mmd_train = evaluate_mmd(real_latents_train, fake_latents, n_samples=mmd_n_samples)
    mmd_test  = evaluate_mmd(real_latents_test,  fake_latents, n_samples=mmd_n_samples)

    w1  = compute_wasserstein_per_feature(
        real_sequences_test, fake_sequences, feature_indices,
        random_state=random_state,
    )
    js  = compute_js_divergence_per_feature(
        real_sequences_test, fake_sequences, feature_indices,
        n_bins=n_bins, random_state=random_state,
    )
    acf = compute_autocorr_fidelity(
        real_sequences_test, fake_sequences, feature_indices,
        max_lag=autocorr_max_lag, random_state=random_state,
    )
    acf_curves = acf.pop("_acf_curves", {})
    phys = physics_pass_rate(fake_sequences, soc_idx=soc_idx, volt_idx=volt_idx)

    metrics = {
        "mmd": {
            "train_split": mmd_train,
            "test_split":  mmd_test,
        },
        "wasserstein_1":          w1,
        "js_divergence":          js,
        "autocorr_fidelity_mad":  acf,
        "acf_curves":             acf_curves,
        "physics_pass_rate":      phys,
    }

    log.info("[EvalSuite] Metric suite complete.")
    return metrics


# ==============================================================================
# Statistical & Temporal Metrics
# ==============================================================================
# Three complementary diagnostics that go beyond marginal distributions:
#
#   1. Sliced Wasserstein Distance (SWD)  — multivariate transport metric that
#      captures joint distributional differences without the curse of
#      dimensionality (Bonneel et al. 2015; Rabin et al. 2012).
#
#   2. ACF-MAE per feature               — temporal-dynamics fidelity: mean
#      absolute error between the real and synthetic autocorrelation functions.
#      Focused on the physics-critical SOC and Voltage channels.
#
#   3. Feature Correlation Matrix Diff   — Frobenius norm ‖C_real − C_synth‖_F
#      over the full 10-feature Pearson correlation matrix, quantifying how
#      well inter-feature relationships are preserved.
#
# All three use held-out test-split arrays only — no data leakage.
# save_statistical_metrics() is the single entry-point called by the pipeline.
#
# References:
#   Bonneel et al. (2015) "Sliced and Radon Wasserstein barycenters". JMIV.
#   Rabin et al. (2012) "Wasserstein barycenter and its application".
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Sliced Wasserstein Distance
# ------------------------------------------------------------------------------

def _project_onto_directions(
    X: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Projects N×D array X onto K random unit vectors, returning N×K matrix."""
    # directions: (K, D) — already unit-normalised by caller
    return X @ directions.T   # (N, K)


def compute_sliced_wasserstein(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    n_projections: int = 200,
    n_samples: Optional[int] = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Computes the **Sliced Wasserstein Distance (SWD)** between real and
    synthetic sequences for each named feature distribution.

    SWD approximates the true multivariate Wasserstein distance by averaging
    the 1-D Wasserstein distances over many random projections.  It is
    computationally tractable for high-dimensional data and avoids the
    histogram-binning artefacts of discrete approximations.

    SWD(P, Q) ≈ (1/K) Σ_k W₁(proj_k(P), proj_k(Q))

    Here each feature is projected across the *time* axis, so sequences of
    shape (N, T) are treated as vectors in ℝᵀ and projected onto K random
    unit directions.  This captures the multivariate temporal distribution
    rather than only the marginal.

    No data leakage: call with held-out test-set sequences.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_indices: Mapping feature_name → column index.
        n_projections:   Number of random projection directions K. Default 200.
        n_samples:       Subsample cap per feature (``None`` = all). Default 2000.
        random_state:    RNG seed. Default 42.

    Returns:
        Dict mapping feature_name → SWD value (float ≥ 0).

    References:
        Bonneel et al. (2015) "Sliced and Radon Wasserstein barycenters of
        measures: A set of numerics". JMIV 51(1):22–45.
    """
    rng = np.random.default_rng(seed=random_state)
    results: Dict[str, float] = {}

    seq_len = real_sequences.shape[1]

    for name, idx in feature_indices.items():
        # Extract time-series vectors: shape (N, seq_len)
        real_ts = real_sequences[:, :, idx].astype(np.float64)
        fake_ts = fake_sequences[:, :, idx].astype(np.float64)

        # Subsample rows if requested
        if n_samples is not None:
            n_r = min(n_samples, len(real_ts))
            n_f = min(n_samples, len(fake_ts))
            real_ts = real_ts[rng.choice(len(real_ts), n_r, replace=False)]
            fake_ts = fake_ts[rng.choice(len(fake_ts), n_f, replace=False)]

        # Generate K random unit directions in ℝ^seq_len
        raw = rng.standard_normal((n_projections, seq_len))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        directions = raw / np.where(norms < 1e-12, 1.0, norms)  # (K, seq_len)

        # Project both sets onto each direction and compute W1
        real_proj = _project_onto_directions(real_ts, directions)  # (N, K)
        fake_proj = _project_onto_directions(fake_ts, directions)  # (M, K)

        w1_per_proj = np.array([
            wasserstein_distance(real_proj[:, k], fake_proj[:, k])
            for k in range(n_projections)
        ])
        swd = float(w1_per_proj.mean())
        results[name] = swd
        log.info(f"[SWD] {name}: SWD={swd:.6f}  (K={n_projections})")

    return results


# ------------------------------------------------------------------------------
# 2. ACF-MAE per named feature (SOC & Voltage focused)
# ------------------------------------------------------------------------------

def compute_acf_mae(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    max_lag: int = 20,
    n_samples: Optional[int] = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Computes the Mean Absolute Error (MAE) between the real and synthetic
    **sample Autocorrelation Functions (ACF)** for each named feature.

    The ACF is averaged across all sequences, then the MAE is computed
    over lags 1 to ``max_lag``.  A low ACF-MAE indicates that the synthetic
    generator faithfully reproduces the temporal autocorrelation structure
    of the real data — a necessary condition for physical plausibility in
    battery charge curves.

    This is the same underlying computation as ``compute_autocorr_fidelity``
    (which is called by ``compute_all_metrics``), exposed here as a named,
    directly-callable function with a user-facing interface oriented around
    the physics-critical features (SOC, Voltage).

    No data leakage: call with held-out test-set sequences.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_indices: Mapping feature_name → column index.
                         Recommended: ``{"SOC": 6, "Voltage": 0}``.
        max_lag:         Maximum ACF lag. Default 20.
        n_samples:       Subsample cap per set (``None`` = all). Default 1000.
        random_state:    RNG seed. Default 42.

    Returns:
        Dict mapping feature_name → ACF-MAE (float ≥ 0).
        Also contains ``"_acf_curves"`` (nested dict of mean ACF arrays per
        feature) for downstream use by ``plot_autocorr_comparison``.
    """
    # Delegate to the existing implementation — avoids code duplication
    result = compute_autocorr_fidelity(
        real_sequences=real_sequences,
        fake_sequences=fake_sequences,
        feature_indices=feature_indices,
        max_lag=max_lag,
        n_samples=n_samples,
        random_state=random_state,
    )
    # Rename the private key to be explicit about its purpose
    acf_curves = result.pop("_acf_curves", {})
    result["acf_curves"] = acf_curves
    return result


# ------------------------------------------------------------------------------
# 3. Feature Correlation Matrix Difference (Frobenius norm)
# ------------------------------------------------------------------------------

def compute_correlation_matrix_diff(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_names: List[str],
    n_samples: Optional[int] = 5000,
    random_state: int = 42,
) -> Dict:
    """
    Computes the Pearson feature correlation matrices for real and synthetic
    sequences and returns the **Frobenius norm** of their difference as a
    scalar fidelity metric.

    ‖C_real − C_synth‖_F = sqrt(Σᵢⱼ (C_real[i,j] − C_synth[i,j])²)

    All time-steps are pooled across sequences before computing correlations
    (i.e., the correlation is over the feature dimension at each timestep,
    averaged across the full dataset).

    No data leakage: call with held-out test-set sequences.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_names:   Ordered list of feature names corresponding to the
                         last axis of the sequence arrays.  Length must equal
                         ``num_features``.  If ``None`` is passed, integer
                         labels ``["f0", "f1", …]`` are used.
        n_samples:       Subsample cap on sequences (``None`` = all). Default 5000.
        random_state:    RNG seed. Default 42.

    Returns:
        Dict with keys:
            ``"frobenius_norm"``   — scalar Frobenius norm of the difference,
            ``"corr_real"``        — 2-D real correlation matrix as nested list,
            ``"corr_synthetic"``   — 2-D synthetic correlation matrix as nested list,
            ``"feature_names"``    — list of feature name strings.
    """
    rng = np.random.default_rng(seed=random_state)

    # Subsample sequences
    if n_samples is not None:
        n_r = min(n_samples, len(real_sequences))
        n_f = min(n_samples, len(fake_sequences))
        real_sequences = real_sequences[rng.choice(len(real_sequences), n_r, replace=False)]
        fake_sequences = fake_sequences[rng.choice(len(fake_sequences), n_f, replace=False)]

    num_features = real_sequences.shape[2]

    # Validate / build feature names
    if feature_names is None or len(feature_names) != num_features:
        if feature_names is not None:
            log.warning(
                f"[CorrMatrix] feature_names length {len(feature_names)} "
                f"!= num_features {num_features}. Using integer labels."
            )
        feature_names = [f"f{i}" for i in range(num_features)]

    # Pool all timesteps: (N*T, F)
    real_flat = real_sequences.reshape(-1, num_features).astype(np.float64)
    fake_flat = fake_sequences.reshape(-1, num_features).astype(np.float64)

    # Remove constant columns (zero variance) to avoid NaN in correlation
    real_var = real_flat.var(axis=0)
    fake_var = fake_flat.var(axis=0)
    valid_mask = (real_var > 1e-10) & (fake_var > 1e-10)

    if not valid_mask.all():
        dropped = [feature_names[i] for i, v in enumerate(valid_mask) if not v]
        log.warning(f"[CorrMatrix] Dropping near-constant features: {dropped}")

    real_valid  = real_flat[:, valid_mask]
    fake_valid  = fake_flat[:, valid_mask]
    valid_names = [feature_names[i] for i, v in enumerate(valid_mask) if v]

    # Pearson correlation matrices
    corr_real  = np.corrcoef(real_valid,  rowvar=False)   # (F', F')
    corr_synth = np.corrcoef(fake_valid,  rowvar=False)

    # Replace any remaining NaN (shouldn't occur after variance filter)
    corr_real  = np.nan_to_num(corr_real,  nan=0.0)
    corr_synth = np.nan_to_num(corr_synth, nan=0.0)

    frob_norm = float(np.linalg.norm(corr_real - corr_synth, ord="fro"))
    log.info(
        f"[CorrMatrix] Frobenius norm ‖C_real − C_synth‖_F = {frob_norm:.6f}  "
        f"({len(valid_names)} features)"
    )

    return {
        "frobenius_norm":  frob_norm,
        "corr_real":       corr_real.tolist(),
        "corr_synthetic":  corr_synth.tolist(),
        "feature_names":   valid_names,
    }


# ------------------------------------------------------------------------------
# Statistical metrics orchestrator + JSON export
# ------------------------------------------------------------------------------

def compute_and_save_statistical_metrics(
    real_sequences_test: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    feature_names: List[str],
    save_path: str,
    soc_volt_indices: Optional[Dict[str, int]] = None,
    n_projections: int = 200,
    max_lag: int = 20,
    n_samples_swd: int = 2000,
    n_samples_acf: int = 1000,
    n_samples_corr: int = 5000,
    random_state: int = 42,
) -> Dict:
    """
    Computes the three statistical/temporal fidelity metrics and persists them
    to ``save_path`` as a JSON file.

    Metrics computed:
        1. **Sliced Wasserstein Distance** (per feature in ``feature_indices``).
        2. **ACF-MAE** for SOC and Voltage (or all features in
           ``soc_volt_indices`` if provided; defaults to ``feature_indices``).
        3. **Feature Correlation Matrix Frobenius norm** over all
           ``num_features`` columns in the sequence arrays.

    All metrics use only held-out test-set arrays — no data leakage.

    Args:
        real_sequences_test: Real test sequences (N_test, seq_len, num_features).
        fake_sequences:      Synthetic sequences (M, seq_len, num_features).
        feature_indices:     Dict mapping feature name → column index.
                             Used for SWD and (if ``soc_volt_indices`` is
                             ``None``) ACF-MAE.
        feature_names:       Ordered list of all feature names (length =
                             num_features).  Used for the correlation matrix.
                             Corresponds to ``DEFAULT_FEATURES`` in
                             ``src/data/sequences.py``.
        save_path:           Output path for ``statistical_metrics.json``.
        soc_volt_indices:    Optional separate dict for ACF-MAE features,
                             e.g. ``{"SOC": 6, "Voltage": 0}``.
                             Defaults to ``feature_indices`` when ``None``.
        n_projections:       Random projections for SWD. Default 200.
        max_lag:             Maximum ACF lag. Default 20.
        n_samples_swd:       Subsample cap for SWD. Default 2000.
        n_samples_acf:       Subsample cap for ACF-MAE. Default 1000.
        n_samples_corr:      Subsample cap for correlation matrix. Default 5000.
        random_state:        Global RNG seed. Default 42.

    Returns:
        The full metrics dict that was saved to JSON.
    """
    log.info("[StatMetrics] Computing statistical & temporal metrics …")

    acf_fi = soc_volt_indices if soc_volt_indices is not None else feature_indices

    # 1. Sliced Wasserstein Distance
    swd = compute_sliced_wasserstein(
        real_sequences_test, fake_sequences,
        feature_indices=feature_indices,
        n_projections=n_projections,
        n_samples=n_samples_swd,
        random_state=random_state,
    )

    # 2. ACF-MAE (SOC + Voltage focus)
    acf_result = compute_acf_mae(
        real_sequences_test, fake_sequences,
        feature_indices=acf_fi,
        max_lag=max_lag,
        n_samples=n_samples_acf,
        random_state=random_state,
    )
    # Separate the curve data from the scalar MAE values for clean JSON output
    acf_curves = acf_result.pop("acf_curves", {})
    acf_mae    = acf_result  # Now only scalar MAD/MAE values keyed by feature name

    # 3. Correlation matrix Frobenius norm
    corr_diff = compute_correlation_matrix_diff(
        real_sequences_test, fake_sequences,
        feature_names=feature_names,
        n_samples=n_samples_corr,
        random_state=random_state,
    )

    metrics = {
        "sliced_wasserstein":          swd,
        "acf_mae":                     acf_mae,
        "acf_curves":                  acf_curves,
        "correlation_matrix_diff": {
            "frobenius_norm":  corr_diff["frobenius_norm"],
            "feature_names":   corr_diff["feature_names"],
        },
        # Store matrices separately (large; useful for heatmap plotting)
        "_corr_real":       corr_diff["corr_real"],
        "_corr_synthetic":  corr_diff["corr_synthetic"],
        "_corr_feature_names": corr_diff["feature_names"],
    }

    save_metrics_json(metrics, save_path=save_path)
    log.info(f"[StatMetrics] Saved → {save_path}")
    return metrics
