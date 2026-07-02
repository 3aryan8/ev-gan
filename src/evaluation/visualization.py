# ==============================================================================
# Evaluation Visualization
# ==============================================================================
# Extracted and improved from 01i / 01k / 01l notebooks.
#
# Rules enforced:
#   - plt.show() is NEVER called; all figures are closed after saving.
#   - Every function accepts a ``save_path`` argument and saves a .pdf.
#   - Feature dimensions are passed as named dicts — no hardcoded indices.
#   - Matplotlib is configured with a non-interactive backend on import.
#   - IEEE formatting: serif font, 10 pt, tight_layout, pdf.fonttype=42.
#   - Wong (2011) colorblind-friendly palette throughout.
#
# Wong (2011) palette (Nature Methods, doi:10.1038/nmeth.1618):
#   Real      → #0072B2  (blue)
#   Synthetic → #E69F00  (orange)
#   Augmented → #009E73  (green)
#   Accent-4  → #CC79A7  (pink)
#   Accent-5  → #56B4E9  (sky blue)
# ==============================================================================

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must precede pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Wong (2011) colorblind-friendly palette
# ------------------------------------------------------------------------------
_REAL_COLOR  = "#0072B2"   # blue
_FAKE_COLOR  = "#E69F00"   # orange
_AUG_COLOR   = "#009E73"   # green  (fine-tuned / augmented)
_ACC4_COLOR  = "#CC79A7"   # pink
_ACC5_COLOR  = "#56B4E9"   # sky blue

_WONG_PALETTE = [_REAL_COLOR, _FAKE_COLOR, _AUG_COLOR, _ACC4_COLOR, _ACC5_COLOR,
                 "#F0E442", "#D55E00", "#000000"]


# ------------------------------------------------------------------------------
# IEEE style helper
# ------------------------------------------------------------------------------

def _apply_ieee_style() -> None:
    """
    Applies IEEE conference-paper rcParams globally:
      - Serif font (Times-compatible), 10 pt body text, 8 pt ticks.
      - Embedded fonts (pdf.fonttype=42) required by IEEE PDF submission.
      - 300 DPI for rasterised elements inside vector PDFs.
      - Clean spines (top/right removed).
    """
    plt.rcParams.update({
        # Typography
        "font.family":        "serif",
        "font.size":          10,
        "axes.titlesize":     10,
        "axes.labelsize":     10,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "figure.titlesize":   11,
        # PDF font embedding (required for IEEE Xplore)
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        # Rasterisation DPI (within vector PDFs)
        "figure.dpi":         300,
        "savefig.dpi":        300,
        # Aesthetics
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
    })


_apply_ieee_style()


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


# ------------------------------------------------------------------------------
# KDE plots
# ------------------------------------------------------------------------------

def plot_kde(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_indices: Dict[str, int],
    save_path: str,
    fig_width: float = 3.5,
    fig_height_per_row: float = 2.0,
) -> None:
    """
    Kernel density estimate (KDE) comparison for named features, overlaying
    real vs. synthetic marginal distributions.

    Each feature gets its own subplot row. The figure is saved as a PDF.

    Args:
        real_sequences:      Real sequence array (N, seq_len, num_features).
        fake_sequences:      Synthetic sequence array (M, seq_len, num_features).
        feature_indices:     Mapping feature_name → column index.
        save_path:           Output PDF path.
        fig_width:           Figure width in inches (3.5 = IEEE single-column).
        fig_height_per_row:  Height per subplot row in inches.
    """
    _ensure_dir(save_path)
    n_features = len(feature_indices)
    fig, axes = plt.subplots(
        n_features, 1,
        figsize=(fig_width, fig_height_per_row * n_features),
        squeeze=False,
    )

    for ax, (name, idx) in zip(axes[:, 0], feature_indices.items()):
        real_vals = real_sequences[:, :, idx].flatten()
        fake_vals = fake_sequences[:, :, idx].flatten()

        sns.kdeplot(real_vals, ax=ax, label="Real",      fill=True, alpha=0.45,
                    color=_REAL_COLOR, linewidth=1.2)
        sns.kdeplot(fake_vals, ax=ax, label="Synthetic", fill=True, alpha=0.45,
                    color=_FAKE_COLOR, linewidth=1.2)
        ax.set_title(f"KDE — {name}")
        ax.set_xlabel("Normalised value")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    fig.suptitle("Real vs. Synthetic: Feature Distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] KDE plot saved → {save_path}")


# ------------------------------------------------------------------------------
# t-SNE plot
# ------------------------------------------------------------------------------

def plot_tsne(
    real_latents: np.ndarray,
    fake_latents: np.ndarray,
    save_path: str,
    n_samples: int = 1000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> None:
    """
    Fits t-SNE on a combined pool of real and synthetic latent vectors and
    saves a 2-D scatter plot comparing the distributions.

    Args:
        real_latents:  Real encoder latent vectors (N, latent_dim).
        fake_latents:  Synthetic generator latent vectors (M, latent_dim).
        save_path:     Output PDF path.
        n_samples:     Samples drawn from each set for t-SNE. Default 1000.
        perplexity:    t-SNE perplexity. Default 30.
        n_iter:        t-SNE iterations. Default 1000.
        random_state:  Reproducibility seed. Default 42.
    """
    _ensure_dir(save_path)
    rng = np.random.default_rng(seed=random_state)
    n = min(n_samples, len(real_latents), len(fake_latents))

    real_sub = real_latents[rng.choice(len(real_latents), n, replace=False)]
    fake_sub = fake_latents[rng.choice(len(fake_latents), n, replace=False)]

    log.info(f"[VIZ] Running t-SNE on {n} real + {n} synthetic latents …")
    X_combined = np.concatenate([real_sub, fake_sub], axis=0)
    tsne = TSNE(
        n_components=2, perplexity=perplexity, n_iter=n_iter,
        random_state=random_state, init="pca",
    )
    X_2d = tsne.fit_transform(X_combined)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(X_2d[:n, 0], X_2d[:n, 1], c=_REAL_COLOR,
               alpha=0.45, s=6, label="Real",      rasterized=True)
    ax.scatter(X_2d[n:, 0], X_2d[n:, 1], c=_FAKE_COLOR,
               alpha=0.45, s=6, label="Synthetic", rasterized=True)

    ax.set_title("t-SNE: Latent Space")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] t-SNE plot saved → {save_path}")


# ------------------------------------------------------------------------------
# PCA plot
# ------------------------------------------------------------------------------

def plot_pca(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    save_path: str,
    n_components: int = 2,
    n_samples: Optional[int] = 2000,
    random_state: int = 42,
) -> None:
    """
    Fits PCA on flattened real + synthetic sequences and saves a 2-D scatter
    plot showing how well synthetic distribution overlaps real.

    Args:
        real_sequences:  Real sequence array (N, seq_len, num_features).
        fake_sequences:  Synthetic sequence array (M, seq_len, num_features).
        save_path:       Output PDF path.
        n_components:    Number of PCA components. Default 2.
        n_samples:       Cap on samples from each set (``None`` = all). Default 2000.
        random_state:    Reproducibility seed. Default 42.
    """
    _ensure_dir(save_path)
    real_flat = real_sequences.reshape(real_sequences.shape[0], -1)
    fake_flat = fake_sequences.reshape(fake_sequences.shape[0], -1)

    if n_samples is not None:
        rng = np.random.default_rng(seed=random_state)
        n_r = min(n_samples, len(real_flat))
        n_f = min(n_samples, len(fake_flat))
        real_flat = real_flat[rng.choice(len(real_flat), n_r, replace=False)]
        fake_flat = fake_flat[rng.choice(len(fake_flat), n_f, replace=False)]

    n_real = len(real_flat)
    X_all  = np.vstack([real_flat, fake_flat])

    log.info(f"[VIZ] Fitting PCA on {len(X_all)} flattened sequences …")
    pca    = PCA(n_components=n_components, random_state=random_state)
    X_proj = pca.fit_transform(X_all)
    var    = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(X_proj[:n_real, 0], X_proj[:n_real, 1],
               c=_REAL_COLOR, alpha=0.4, s=6, label="Real",      rasterized=True)
    ax.scatter(X_proj[n_real:, 0], X_proj[n_real:, 1],
               c=_FAKE_COLOR, alpha=0.4, s=6, label="Synthetic", rasterized=True)

    ax.set_title("PCA: Sequence Space")
    ax.set_xlabel(f"PC 1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({var[1]:.1f}% var)")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] PCA plot saved → {save_path}")


# ------------------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------------------

def plot_training_curves(
    ae_history_df: pd.DataFrame,
    save_path: str,
    gan_history_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Two-panel (or single-panel) training curve figure.

    Left panel: Autoencoder train/val loss vs. epoch.
    Right panel (if provided): GAN generator and discriminator losses vs. epoch.

    Args:
        ae_history_df:  DataFrame with columns ``["loss", "val_loss"]`` and
                        integer row index (epoch). Produced by
                        ``pd.DataFrame(autoencoder.fit(...).history)``.
        save_path:      Output PDF path.
        gan_history_df: Optional DataFrame with columns ``["g_loss", "d_loss"]``.
    """
    _ensure_dir(save_path)
    n_panels = 2 if gan_history_df is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 2.8))
    if n_panels == 1:
        axes = [axes]

    # AE panel
    ax = axes[0]
    epochs = range(1, len(ae_history_df) + 1)
    ax.plot(epochs, ae_history_df["loss"],     color=_REAL_COLOR,
            linewidth=1.2, label="Train loss")
    if "val_loss" in ae_history_df.columns:
        ax.plot(epochs, ae_history_df["val_loss"], color=_FAKE_COLOR,
                linewidth=1.2, linestyle="--", label="Val loss")
    ax.set_title("Autoencoder Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(frameon=False)

    # GAN panel (optional)
    if gan_history_df is not None:
        ax2 = axes[1]
        g_epochs = range(1, len(gan_history_df) + 1)
        if "g_loss" in gan_history_df.columns:
            ax2.plot(g_epochs, gan_history_df["g_loss"], color=_REAL_COLOR,
                     linewidth=1.0, label="G loss")
        if "d_loss" in gan_history_df.columns:
            ax2.plot(g_epochs, gan_history_df["d_loss"], color=_FAKE_COLOR,
                     linewidth=1.0, linestyle="--", label="D loss (W)")
        ax2.set_title("WGAN-GP Training")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] Training curves saved → {save_path}")


# ------------------------------------------------------------------------------
# SOH comparison bar chart
# ------------------------------------------------------------------------------

def plot_soh_comparison(
    soh_metrics: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """
    Grouped bar chart comparing Baseline vs. Pretrained SOH regression.

    Two metric groups are displayed side by side: RMSE and MAE.

    Args:
        soh_metrics: Dict as returned by ``run_downstream_soh()``, e.g.
                     ``{"baseline": {"rmse": 0.04, "mae": 0.03},
                        "pretrained": {"rmse": 0.038, "mae": 0.028}}``.
        save_path:   Output PDF path.
    """
    _ensure_dir(save_path)

    labels  = ["RMSE", "MAE"]
    base    = [soh_metrics["baseline"]["rmse"],   soh_metrics["baseline"]["mae"]]
    pretrained = [soh_metrics["pretrained"]["rmse"], soh_metrics["pretrained"]["mae"]]

    x   = np.arange(len(labels))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    bars_b = ax.bar(x - w / 2, base,      width=w, color=_REAL_COLOR, label="Baseline",   alpha=0.85)
    bars_p = ax.bar(x + w / 2, pretrained, width=w, color=_AUG_COLOR,  label="Pretrained", alpha=0.85)

    # Annotate bar tops
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (normalised units)")
    ax.set_title("SOH Prediction: Baseline vs. Pretrained")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] SOH comparison saved → {save_path}")


# ------------------------------------------------------------------------------
# Physics pass-rate plot
# ------------------------------------------------------------------------------

def plot_physics_pass_rate(
    pass_rate_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """
    Horizontal bar chart showing per-rule and combined physics pass rates.

    Args:
        pass_rate_metrics: Dict as returned by ``physics_pass_rate()``, with
                           keys ``"rule_soc_monotonicity"``, ``"rule_soc_variance"``,
                           ``"rule_volt_stability"``, ``"combined"``.
        save_path:         Output PDF path.
    """
    _ensure_dir(save_path)

    rule_map = {
        "SOC Monotonicity":  pass_rate_metrics.get("rule_soc_monotonicity", 0.0),
        "SOC Variance":      pass_rate_metrics.get("rule_soc_variance",     0.0),
        "Volt. Stability":   pass_rate_metrics.get("rule_volt_stability",   0.0),
        "Combined":          pass_rate_metrics.get("combined",              0.0),
    }
    names  = list(rule_map.keys())
    values = [v * 100 for v in rule_map.values()]
    colors = [_ACC5_COLOR, _ACC5_COLOR, _ACC5_COLOR, _AUG_COLOR]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(min(val + 1.0, 99.0), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Pass rate (%)")
    ax.set_title("Physics Constraint Pass Rates")
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] Physics pass-rate plot saved → {save_path}")


# ------------------------------------------------------------------------------
# Autocorrelation comparison
# ------------------------------------------------------------------------------

def plot_autocorr_comparison(
    acf_curves: Dict[str, Dict[str, List[float]]],
    save_path: str,
    fig_width: float = 3.5,
    fig_height_per_row: float = 2.0,
) -> None:
    """
    Multi-panel autocorrelation comparison: mean ACF vs. lag for real and
    synthetic sequences, one subplot per feature.

    Args:
        acf_curves:          Dict of feature_name → ``{"real": [...], "fake": [...]}``.
                             This is the ``"_acf_curves"`` field from
                             ``compute_autocorr_fidelity()``.
        save_path:           Output PDF path.
        fig_width:           Figure width in inches.
        fig_height_per_row:  Height per subplot in inches.
    """
    _ensure_dir(save_path)
    n = len(acf_curves)
    if n == 0:
        log.warning("[VIZ] plot_autocorr_comparison: no ACF curves provided.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height_per_row * n), squeeze=False)

    for ax, (name, curves) in zip(axes[:, 0], acf_curves.items()):
        lags = range(1, len(curves["real"]) + 1)
        ax.plot(lags, curves["real"], color=_REAL_COLOR,  linewidth=1.2, label="Real")
        ax.plot(lags, curves["fake"], color=_FAKE_COLOR,  linewidth=1.2,
                linestyle="--", label="Synthetic")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_title(f"Autocorrelation — {name}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] Autocorrelation plot saved → {save_path}")


# ------------------------------------------------------------------------------
# Sequence sample panel
# ------------------------------------------------------------------------------

def plot_sequence_samples(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    feature_idx: int,
    feature_name: str,
    save_path: str,
    n_samples: int = 8,
    random_state: int = 42,
) -> None:
    """
    Grid of N randomly sampled real vs. synthetic time-series for a single
    feature, arranged in two columns (Real | Synthetic) for visual inspection
    of waveform quality.

    Args:
        real_sequences:  Real sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic sequences (M, seq_len, num_features).
        feature_idx:     Column index of the feature to plot.
        feature_name:    Human-readable feature name (for titles).
        save_path:       Output PDF path.
        n_samples:       Number of sample pairs to draw. Default 8.
        random_state:    RNG seed. Default 42.
    """
    _ensure_dir(save_path)
    rng   = np.random.default_rng(seed=random_state)
    n_r   = min(n_samples, len(real_sequences))
    n_f   = min(n_samples, len(fake_sequences))
    n_plt = min(n_r, n_f)

    real_idx = rng.choice(len(real_sequences), n_plt, replace=False)
    fake_idx = rng.choice(len(fake_sequences), n_plt, replace=False)

    fig, axes = plt.subplots(
        n_plt, 2,
        figsize=(3.5, 1.4 * n_plt),
        sharex=False, sharey=True,
    )
    # Guard single-row case
    if n_plt == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Real",      fontsize=9)
    axes[0, 1].set_title("Synthetic", fontsize=9)

    for row in range(n_plt):
        r_seq = real_sequences[real_idx[row], :, feature_idx]
        f_seq = fake_sequences[fake_idx[row], :, feature_idx]

        axes[row, 0].plot(r_seq, color=_REAL_COLOR,  linewidth=0.8)
        axes[row, 1].plot(f_seq, color=_FAKE_COLOR,  linewidth=0.8)

        for col in range(2):
            axes[row, col].tick_params(labelsize=6)
            axes[row, col].set_yticks([])

    fig.suptitle(f"Sample Sequences — {feature_name}", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] Sequence sample panel saved → {save_path}")


# ------------------------------------------------------------------------------
# Convenience: run all plots in one call
# ------------------------------------------------------------------------------

def plot_all(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    real_latents: np.ndarray,
    fake_latents: np.ndarray,
    feature_indices: Dict[str, int],
    figures_dir: str = "artifacts/figures",
    random_state: int = 42,
    soh_metrics: Optional[Dict] = None,
    pass_rate_metrics: Optional[Dict] = None,
    acf_curves: Optional[Dict] = None,
    ae_history_df: Optional[pd.DataFrame] = None,
    gan_history_df: Optional[pd.DataFrame] = None,
    sequence_sample_feature: Optional[str] = None,
    n_sequence_samples: int = 8,
) -> None:
    """
    Generates and saves all diagnostic plots to ``figures_dir`` as PDFs.

    Always generated:
        - kde_comparison.pdf
        - tsne_latent.pdf
        - pca_sequences.pdf

    Generated when optional data is supplied:
        - training_curves.pdf     (requires ae_history_df)
        - soh_comparison.pdf      (requires soh_metrics)
        - physics_pass_rate.pdf   (requires pass_rate_metrics)
        - autocorr_comparison.pdf (requires acf_curves)
        - sequence_samples.pdf    (requires sequence_sample_feature name)

    Args:
        real_sequences:           Real decoded sequences (N, seq_len, num_features).
        fake_sequences:           Synthetic decoded sequences (M, seq_len, num_features).
        real_latents:             Real encoder latent vectors (N, latent_dim).
        fake_latents:             Synthetic generator latent vectors (M, latent_dim).
        feature_indices:          Dict mapping feature name → column index for KDE/ACF.
        figures_dir:              Output directory. Created if absent.
        random_state:             Seed for t-SNE / PCA subsampling.
        soh_metrics:              Optional dict from ``run_downstream_soh()``.
        pass_rate_metrics:        Optional dict from ``physics_pass_rate()``.
        acf_curves:               Optional dict from ``compute_autocorr_fidelity()``.
        ae_history_df:            Optional AE training history DataFrame.
        gan_history_df:           Optional GAN training history DataFrame.
        sequence_sample_feature:  Feature name in ``feature_indices`` to use for
                                  the sample panel (``None`` skips the plot).
        n_sequence_samples:       Number of sequence pairs in the sample panel.
    """
    os.makedirs(figures_dir, exist_ok=True)

    plot_kde(
        real_sequences, fake_sequences,
        feature_indices=feature_indices,
        save_path=os.path.join(figures_dir, "kde_comparison.pdf"),
    )
    plot_tsne(
        real_latents, fake_latents,
        save_path=os.path.join(figures_dir, "tsne_latent.pdf"),
        random_state=random_state,
    )
    plot_pca(
        real_sequences, fake_sequences,
        save_path=os.path.join(figures_dir, "pca_sequences.pdf"),
        random_state=random_state,
    )

    if ae_history_df is not None:
        plot_training_curves(
            ae_history_df=ae_history_df,
            gan_history_df=gan_history_df,
            save_path=os.path.join(figures_dir, "training_curves.pdf"),
        )

    if soh_metrics is not None:
        plot_soh_comparison(
            soh_metrics=soh_metrics,
            save_path=os.path.join(figures_dir, "soh_comparison.pdf"),
        )

    if pass_rate_metrics is not None:
        plot_physics_pass_rate(
            pass_rate_metrics=pass_rate_metrics,
            save_path=os.path.join(figures_dir, "physics_pass_rate.pdf"),
        )

    if acf_curves is not None:
        plot_autocorr_comparison(
            acf_curves=acf_curves,
            save_path=os.path.join(figures_dir, "autocorr_comparison.pdf"),
        )

    if sequence_sample_feature is not None and sequence_sample_feature in feature_indices:
        plot_sequence_samples(
            real_sequences=real_sequences,
            fake_sequences=fake_sequences,
            feature_idx=feature_indices[sequence_sample_feature],
            feature_name=sequence_sample_feature,
            save_path=os.path.join(figures_dir, "sequence_samples.pdf"),
            n_samples=n_sequence_samples,
            random_state=random_state,
        )


# ------------------------------------------------------------------------------
# Correlation heatmaps (Real vs. Synthetic)
# ------------------------------------------------------------------------------

def plot_correlation_heatmaps(
    corr_real: np.ndarray,
    corr_synthetic: np.ndarray,
    feature_names: List[str],
    save_path: str,
    frobenius_norm: Optional[float] = None,
) -> None:
    """
    1×2 subplot figure showing the Pearson feature correlation matrices for
    real and synthetic sequences side by side as annotated heatmaps.

    Design choices (IEEE-compliant):
      - Diverging palette ``RdBu_r`` centred at 0 (correlations ∈ [−1, 1]).
      - Shared colour scale (``vmin=−1, vmax=1``) so both panels are directly
        comparable.
      - Single shared colour-bar on the right.
      - Frobenius norm ``‖C_real − C_synth‖_F`` displayed in the figure
        super-title when provided.
      - Cell values annotated to two decimal places; font size 6 pt to fit
        within IEEE single-column width at any reasonable feature count.
      - Saved as PDF with embedded fonts (``pdf.fonttype=42``).

    Args:
        corr_real:      Pearson correlation matrix for real data, shape (F, F).
                        May be a NumPy array or a nested list (as returned by
                        ``compute_correlation_matrix_diff``).
        corr_synthetic: Pearson correlation matrix for synthetic data, shape (F, F).
        feature_names:  List of F feature name strings for axis tick labels.
        save_path:      Output PDF path (e.g., ``artifacts/figures/eval_correlation_heatmap.pdf``).
        frobenius_norm: Optional scalar ``‖C_real − C_synth‖_F`` to annotate
                        in the super-title.
    """
    _ensure_dir(save_path)

    corr_real      = np.asarray(corr_real,      dtype=float)
    corr_synthetic = np.asarray(corr_synthetic, dtype=float)
    n_feat         = len(feature_names)

    # Scale figure width: IEEE double-column (7.16 in) for ≥6 features,
    # single-column (3.5 in) for fewer
    fig_w = 7.16 if n_feat >= 6 else 5.0
    fig_h = max(2.8, 0.45 * n_feat + 1.2)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # Shared diverging palette centred at zero
    cmap   = "RdBu_r"
    vmin, vmax = -1.0, 1.0

    # Short display names (truncate long feature names to 12 chars)
    short_names = [n[:12] for n in feature_names]

    # Annotation font size: shrink for large matrices
    annot_kw = {"size": max(4, 7 - n_feat // 4)}

    for ax, matrix, title in zip(
        axes,
        [corr_real, corr_synthetic],
        ["Real", "Synthetic"],
    ):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws=annot_kw,
            linewidths=0.3,
            linecolor="white",
            xticklabels=short_names,
            yticklabels=short_names,
            cbar=(ax is axes[-1]),         # colour-bar only on the right panel
            square=True,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

    # Super-title with Frobenius norm
    if frobenius_norm is not None:
        fig.suptitle(
            f"Feature Correlation Matrix — Real vs. Synthetic\n"
            f"‖C_real − C_synth‖_F = {frobenius_norm:.4f}",
            y=1.02,
        )
    else:
        fig.suptitle("Feature Correlation Matrix — Real vs. Synthetic", y=1.02)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] Correlation heatmaps saved → {save_path}")
