# ==============================================================================
# Evaluation Visualization
# ==============================================================================
# Extracted and improved from 01i / 01k / 01l notebooks.
#
# Rules enforced:
#   - plt.show() is NEVER called; all figures are closed after saving.
#   - Every function accepts a ``save_path`` argument and saves a .pdf.
#   - Feature dimensions are passed as named dicts — no hardcoded indices.
#   - Matplotlib is configured with a non-interactive backend on import to
#     prevent accidental display in headless environments.
# ==============================================================================

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must precede pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)

# Shared style applied to every figure
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_REAL_COLOR = "#4C72B0"
_FAKE_COLOR = "#DD8452"


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
    fig_width: float = 10.0,
    fig_height_per_row: float = 3.0,
) -> None:
    """
    Generates kernel density estimate (KDE) comparison plots for a set of
    features, overlaying real vs. synthetic distributions.

    Each feature gets its own subplot row.  The figure is saved as a PDF at
    ``save_path`` and the figure is immediately closed — ``plt.show()`` is
    never called.

    Args:
        real_sequences:  Real sequence array (N, seq_len, num_features).
        fake_sequences:  Synthetic sequence array (M, seq_len, num_features).
        feature_indices: Mapping of human-readable name → column index, e.g.
                         ``{"SOC": 6, "Voltage": 0, "Current": 1}``.
        save_path:       Absolute or relative path for the output PDF.
        fig_width:       Figure width in inches.
        fig_height_per_row: Height per subplot row in inches.
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

        sns.kdeplot(real_vals, ax=ax, label="Real", fill=True, alpha=0.5,
                    color=_REAL_COLOR)
        sns.kdeplot(fake_vals, ax=ax, label="Synthetic", fill=True, alpha=0.5,
                    color=_FAKE_COLOR)
        ax.set_title(f"KDE — {name}", fontsize=11)
        ax.set_xlabel("Normalised value")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    fig.suptitle("Real vs Synthetic: Feature Distributions", fontsize=13, y=1.01)
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
        save_path:     Absolute or relative path for the output PDF.
        n_samples:     Number of samples drawn from each set for t-SNE.
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
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    X_2d = tsne.fit_transform(X_combined)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_2d[:n, 0], X_2d[:n, 1], c=_REAL_COLOR,
               alpha=0.45, s=12, label="Real", rasterized=True)
    ax.scatter(X_2d[n:, 0], X_2d[n:, 1], c=_FAKE_COLOR,
               alpha=0.45, s=12, label="Synthetic", rasterized=True)

    ax.set_title("t-SNE: Latent Space — Real vs Synthetic", fontsize=12)
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
    plot showing how well the synthetic distribution overlaps the real one.

    Sequences are flattened along the (seq_len × num_features) dimension before
    PCA, matching the approach in the notebooks.

    Args:
        real_sequences:  Real sequence array (N, seq_len, num_features).
        fake_sequences:  Synthetic sequence array (M, seq_len, num_features).
        save_path:       Absolute or relative path for the output PDF.
        n_components:    Number of PCA components. Default 2.
        n_samples:       Cap on samples from each set (``None`` = all).
        random_state:    Reproducibility seed for subsetting. Default 42.
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
    X_all = np.vstack([real_flat, fake_flat])

    log.info(f"[VIZ] Fitting PCA on {len(X_all)} flattened sequences …")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_proj = pca.fit_transform(X_all)

    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_proj[:n_real, 0], X_proj[:n_real, 1],
               c=_REAL_COLOR, alpha=0.4, s=10, label="Real", rasterized=True)
    ax.scatter(X_proj[n_real:, 0], X_proj[n_real:, 1],
               c=_FAKE_COLOR, alpha=0.4, s=10, label="Synthetic", rasterized=True)

    ax.set_title("PCA: Sequence Space — Real vs Synthetic", fontsize=12)
    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({var_explained[1]:.1f}% var)")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"[VIZ] PCA plot saved → {save_path}")


# ------------------------------------------------------------------------------
# Convenience: run all three plots in one call
# ------------------------------------------------------------------------------

def plot_all(
    real_sequences: np.ndarray,
    fake_sequences: np.ndarray,
    real_latents: np.ndarray,
    fake_latents: np.ndarray,
    feature_indices: Dict[str, int],
    figures_dir: str = "artifacts/figures",
    random_state: int = 42,
) -> None:
    """
    Generates and saves all three diagnostic plots (KDE, t-SNE, PCA) to
    ``figures_dir`` as PDF files.

    Args:
        real_sequences:  Real decoded sequences (N, seq_len, num_features).
        fake_sequences:  Synthetic decoded sequences (M, seq_len, num_features).
        real_latents:    Real encoder latent vectors (N, latent_dim).
        fake_latents:    Synthetic generator latent vectors (M, latent_dim).
        feature_indices: Dict mapping feature name → column index for KDE.
        figures_dir:     Output directory. Created if absent.
        random_state:    Seed for t-SNE and PCA subsampling.
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
