#!/usr/bin/env python
# ==============================================================================
# generate_figures.py — Standalone Figure Regeneration Script
# ==============================================================================
# Loads pre-saved .npy artifacts and metrics JSON produced by run_pipeline.py,
# then regenerates the full IEEE-formatted figure suite without re-running
# training or inference.
#
# Usage (from repo root):
#   PYTHONPATH=. python scripts/generate_figures.py
#   PYTHONPATH=. python scripts/generate_figures.py --figures-dir artifacts/figures_v2
#   PYTHONPATH=. python scripts/generate_figures.py --config artifacts/run_config.yaml
#   PYTHONPATH=. python scripts/generate_figures.py --dry-run
#
# Artifact dependencies (all under repo root unless overridden):
#   data/processed/test_sequences.npy
#   data/processed/test_conditioning.npy
#   data/processed/train_latents.npy       (for backward-compat MMD display)
#   data/synthetic/synthetic_filtered.npy  (falls back to synthetic_sequences.npy)
#   data/synthetic/synthetic_latents.npy
#   artifacts/metrics/eval_metrics.json    (optional — metrics printed if present)
#   artifacts/metrics/ae_history.csv       (optional — training curves)
#   artifacts/metrics/gan_history.csv      (optional — GAN curves)
# ==============================================================================

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

# Configure logging before any src imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_figures")


# ------------------------------------------------------------------------------
# Path defaults
# ------------------------------------------------------------------------------

_DEFAULTS = {
    "processed_dir": "data/processed",
    "synthetic_dir": "data/synthetic",
    "metrics_dir":   "artifacts/metrics",
    "figures_dir":   "artifacts/figures",
    "config_path":   "artifacts/run_config.yaml",
}

_REQUIRED_ARTIFACTS = [
    ("data/processed/test_sequences.npy",      "Real test sequences"),
    ("data/processed/test_conditioning.npy",   "Real test conditions"),
    ("data/processed/train_latents.npy",       "Train encoder latents"),
    ("data/synthetic/synthetic_latents.npy",   "Synthetic generator latents"),
]

_OPTIONAL_ARTIFACTS = [
    ("data/synthetic/synthetic_filtered.npy",  "Physics-filtered synthetic sequences"),
    ("data/synthetic/synthetic_sequences.npy", "Unfiltered synthetic sequences"),
    ("artifacts/metrics/eval_metrics.json",    "Evaluation metrics JSON"),
    ("artifacts/metrics/ae_history.csv",       "Autoencoder training history"),
    ("artifacts/metrics/gan_history.csv",      "GAN training history"),
]


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenerate IEEE-formatted figures from saved PIC-GAN artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--figures-dir", default=_DEFAULTS["figures_dir"],
        help="Output directory for generated PDF figures.",
    )
    p.add_argument(
        "--config", default=_DEFAULTS["config_path"],
        help="Path to run_config.yaml snapshot (used for feature_indices, etc.).",
    )
    p.add_argument(
        "--processed-dir", default=_DEFAULTS["processed_dir"],
        help="Directory containing processed .npy arrays.",
    )
    p.add_argument(
        "--synthetic-dir", default=_DEFAULTS["synthetic_dir"],
        help="Directory containing synthetic .npy arrays.",
    )
    p.add_argument(
        "--metrics-dir", default=_DEFAULTS["metrics_dir"],
        help="Directory containing metrics JSON and history CSVs.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate that all required artifact paths exist and exit without generating figures.",
    )
    p.add_argument(
        "--n-sequence-samples", type=int, default=8,
        help="Number of sample pairs in the sequence panel plot.",
    )
    p.add_argument(
        "--sequence-feature", default="SOC",
        help="Feature name to use for the sequence sample panel.",
    )
    return p


# ------------------------------------------------------------------------------
# Artifact validation
# ------------------------------------------------------------------------------

def _validate_artifacts(required: list, optional: list, dry_run: bool) -> bool:
    """
    Checks that all required artifacts exist.  Reports optional missing files
    as warnings.  In dry-run mode, prints a full manifest and exits.

    Returns True if all required artifacts are present.
    """
    ok = True
    log.info("=" * 60)
    log.info("  ARTIFACT MANIFEST")
    log.info("=" * 60)

    for path, label in required:
        exists = os.path.isfile(path)
        status = "✓" if exists else "✗ MISSING"
        log.info(f"  [{status}] {label}")
        log.info(f"          {path}")
        if not exists:
            ok = False

    for path, label in optional:
        exists = os.path.isfile(path)
        status = "✓" if exists else "- (optional)"
        log.info(f"  [{status}] {label}")
        log.info(f"          {path}")

    log.info("=" * 60)

    if not ok:
        log.error(
            "One or more REQUIRED artifacts are missing. "
            "Run `scripts/run_pipeline.py` first to generate them."
        )
    elif dry_run:
        log.info("Dry-run complete — all required artifacts present.")
    return ok


# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads feature_indices and evaluation settings from a run_config.yaml
    snapshot using OmegaConf (if available) or PyYAML as fallback.

    Returns a plain dict (never an OmegaConf DictConfig) so callers
    don't need omegaconf as a hard dependency.
    """
    if not os.path.isfile(config_path):
        log.warning(
            f"Config not found at '{config_path}'. "
            "Using default feature indices: volt=0, curr=1, soc=6, temp=4."
        )
        return {
            "feature_indices": {"volt": 0, "curr": 1, "soc": 6, "temp": 4},
            "evaluation": {"n_bins": 50, "autocorr_max_lag": 20,
                           "n_samples_viz": 1000, "n_sequence_samples": 8},
        }

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        log.info(f"Config loaded via OmegaConf: {config_path}")
        return cfg
    except ImportError:
        pass

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    log.info(f"Config loaded via PyYAML: {config_path}")
    return cfg


# ------------------------------------------------------------------------------
# Metrics display helper
# ------------------------------------------------------------------------------

def _print_metrics_summary(metrics_path: str) -> Optional[Dict]:
    """Reads and pretty-prints the eval_metrics.json if it exists."""
    if not os.path.isfile(metrics_path):
        log.info("No eval_metrics.json found — skipping metrics summary.")
        return None

    with open(metrics_path) as f:
        data = json.load(f)

    m = data.get("metrics", {})
    log.info("")
    log.info("  METRIC SUMMARY")
    log.info("-" * 48)

    # MMD
    mmd = m.get("mmd", {})
    for split, vals in mmd.items():
        if isinstance(vals, dict):
            log.info(f"  MMD² ({split}): {vals.get('mmd2', 'N/A'):.6f}  "
                     f"γ={vals.get('gamma', 'N/A'):.6f}")

    # Per-feature scalars
    for metric_key in ("wasserstein_1", "js_divergence", "autocorr_fidelity_mad"):
        feat_dict = m.get(metric_key, {})
        if feat_dict:
            log.info(f"\n  {metric_key}:")
            for feat, val in feat_dict.items():
                if isinstance(val, float):
                    log.info(f"    {feat}: {val:.6f}")

    # Physics pass rate
    phys = m.get("physics_pass_rate", {})
    if phys:
        log.info(f"\n  Physics pass rate: {phys.get('combined', 0.0)*100:.1f}% combined  "
                 f"(n={int(phys.get('n_total', 0))})")

    log.info("-" * 48)
    return m


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Build artifact path lists using resolved dirs
    proc = args.processed_dir
    syn  = args.synthetic_dir
    met  = args.metrics_dir

    required = [
        (os.path.join(proc, "test_sequences.npy"),    "Real test sequences"),
        (os.path.join(proc, "test_conditioning.npy"), "Real test conditions"),
        (os.path.join(proc, "train_latents.npy"),     "Train encoder latents"),
        (os.path.join(syn,  "synthetic_latents.npy"), "Synthetic generator latents"),
    ]
    optional = [
        (os.path.join(syn, "synthetic_filtered.npy"),     "Physics-filtered synthetic seqs"),
        (os.path.join(syn, "synthetic_sequences.npy"),    "Unfiltered synthetic seqs"),
        (os.path.join(met, "eval_metrics.json"),          "Evaluation metrics JSON"),
        (os.path.join(met, "ae_history.csv"),             "Autoencoder training history"),
        (os.path.join(met, "gan_history.csv"),            "GAN training history"),
    ]

    ok = _validate_artifacts(required, optional, dry_run=args.dry_run)
    if not ok or args.dry_run:
        sys.exit(0 if (args.dry_run and ok) else 1)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = _load_config(args.config)
    fi_raw = cfg.get("feature_indices", {"volt": 0, "curr": 1, "soc": 6, "temp": 4})
    eval_cfg = cfg.get("evaluation", {})

    soc_idx  = int(fi_raw.get("soc",  6))
    volt_idx = int(fi_raw.get("volt", 0))
    curr_idx = int(fi_raw.get("curr", 1))

    kde_features = {"SOC": soc_idx, "Voltage": volt_idx, "Current": curr_idx}

    # ------------------------------------------------------------------
    # Load .npy arrays
    # ------------------------------------------------------------------
    log.info("Loading .npy arrays …")
    X_test        = np.load(os.path.join(proc, "test_sequences.npy"))
    train_latents = np.load(os.path.join(proc, "train_latents.npy"))
    syn_latents   = np.load(os.path.join(syn,  "synthetic_latents.npy"))

    # Prefer filtered synthetic; fall back to unfiltered
    filtered_path   = os.path.join(syn, "synthetic_filtered.npy")
    unfiltered_path = os.path.join(syn, "synthetic_sequences.npy")
    if os.path.isfile(filtered_path):
        syn_sequences = np.load(filtered_path)
        log.info(f"Using physics-filtered synthetic sequences: {syn_sequences.shape}")
    elif os.path.isfile(unfiltered_path):
        syn_sequences = np.load(unfiltered_path)
        log.warning("Physics-filtered sequences not found — using unfiltered.")
    else:
        log.error("No synthetic sequence arrays found. Exiting.")
        sys.exit(1)

    # test_latents: prefer test_latents.npy; fall back to train_latents for display
    test_latents_path = os.path.join(proc, "test_latents.npy")
    if os.path.isfile(test_latents_path):
        test_latents = np.load(test_latents_path)
    else:
        log.warning("test_latents.npy not found — using train_latents for latent plots.")
        test_latents = train_latents

    log.info(
        f"Arrays loaded — X_test: {X_test.shape}  "
        f"syn_seqs: {syn_sequences.shape}  "
        f"latents: {test_latents.shape}"
    )

    # ------------------------------------------------------------------
    # Optional: load history DataFrames
    # ------------------------------------------------------------------
    import pandas as pd

    ae_history_df  = None
    gan_history_df = None

    ae_hist_path  = os.path.join(met, "ae_history.csv")
    gan_hist_path = os.path.join(met, "gan_history.csv")

    if os.path.isfile(ae_hist_path):
        ae_history_df = pd.read_csv(ae_hist_path)
        log.info(f"AE history loaded: {ae_history_df.shape[0]} epochs")
    if os.path.isfile(gan_hist_path):
        gan_history_df = pd.read_csv(gan_hist_path)
        log.info(f"GAN history loaded: {gan_history_df.shape[0]} epochs")

    # ------------------------------------------------------------------
    # Optional: load eval_metrics.json for SOH, pass-rate, ACF curves
    # ------------------------------------------------------------------
    metrics_path = os.path.join(met, "eval_metrics.json")
    saved_metrics = _print_metrics_summary(metrics_path)

    soh_metrics       = None
    pass_rate_metrics = None
    acf_curves        = None

    if saved_metrics:
        soh_metrics       = saved_metrics.get("soh")
        pass_rate_metrics = saved_metrics.get("physics_pass_rate")
        acf_curves        = saved_metrics.get("acf_curves")

    # ------------------------------------------------------------------
    # Generate all figures
    # ------------------------------------------------------------------
    log.info(f"Generating figures → {args.figures_dir}")

    from src.evaluation.visualization import plot_all

    plot_all(
        real_sequences=X_test,
        fake_sequences=syn_sequences,
        real_latents=test_latents,
        fake_latents=syn_latents,
        feature_indices=kde_features,
        figures_dir=args.figures_dir,
        soh_metrics=soh_metrics,
        pass_rate_metrics=pass_rate_metrics,
        acf_curves=acf_curves,
        ae_history_df=ae_history_df,
        gan_history_df=gan_history_df,
        sequence_sample_feature=args.sequence_feature
            if args.sequence_feature in kde_features else None,
        n_sequence_samples=args.n_sequence_samples,
    )

    log.info("")
    log.info("=" * 60)
    log.info(f"  Figures written to: {args.figures_dir}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
