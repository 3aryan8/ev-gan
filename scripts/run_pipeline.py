#!/usr/bin/env python
# ==============================================================================
# PIC-GAN End-to-End Pipeline Entry Point
# ==============================================================================
# Usage (from repo root):
#   PYTHONPATH=. python scripts/run_pipeline.py
#   PYTHONPATH=. python scripts/run_pipeline.py training.epochs_gan=500 model.latent_dim=64
#   PYTHONPATH=. python scripts/run_pipeline.py physics_mode=advanced
#
# The exact resolved config is saved to artifacts/run_config.yaml before any
# computation begins, guaranteeing full experiment reproducibility.
#
# Pipeline stages (in order):
#   1. Seed global RNGs
#   2. Copy resolved config  → artifacts/run_config.yaml
#   3. Data prep             → data/processed/  +  artifacts/models/scaler.pkl
#   4. Autoencoder training  → artifacts/models/{encoder,decoder}.keras
#   5. WGAN-GP training      → artifacts/models/{generator,discriminator}.keras
#   6. Synthetic generation  → data/synthetic/
#   7. Evaluation (MMD + plots) → artifacts/metrics/ + artifacts/figures/
#   8. Downstream SOH task   → artifacts/metrics/results.csv + artifacts/tables/results.tex
# ==============================================================================

import logging
import os
import shutil
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _stage(name: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info(f"  STAGE: {name}")
    log.info("=" * 68)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_start = time.time()

    # ------------------------------------------------------------------
    # 0. Seed + config snapshot
    # ------------------------------------------------------------------
    _stage("0 — Seeding & Config Snapshot")

    from src.utils.seeding import set_seed
    set_seed(int(cfg.seed))

    # Persist the fully-resolved config before touching any data so the
    # experiment can be reproduced exactly from this file alone.
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    config_snapshot_path = os.path.join(artifacts_dir, "run_config.yaml")
    OmegaConf.save(cfg, config_snapshot_path)
    log.info(f"Config snapshot saved → {config_snapshot_path}")

    # ------------------------------------------------------------------
    # Resolve commonly used config values once
    # ------------------------------------------------------------------
    seed          = int(cfg.seed)
    seq_len       = int(cfg.model.sequence_length)
    latent_dim    = int(cfg.model.latent_dim)
    noise_dim     = int(cfg.model.noise_dim)
    lstm_units    = int(cfg.model.lstm_units)
    batch_size    = int(cfg.training.batch_size)

    processed_dir = "data/processed"
    synthetic_dir = "data/synthetic"
    models_dir    = "artifacts/models"
    figures_dir   = "artifacts/figures"
    metrics_dir   = "artifacts/metrics"
    tables_dir    = "artifacts/tables"
    scaler_path   = os.path.join(models_dir, "scaler.pkl")

    for d in (processed_dir, synthetic_dir, models_dir, figures_dir,
              metrics_dir, tables_dir):
        os.makedirs(d, exist_ok=True)

    # Feature indices — single source of truth from config
    fi = cfg.feature_indices
    soc_idx  = int(fi.soc)
    volt_idx = int(fi.volt)
    curr_idx = int(fi.curr)
    temp_idx = int(fi.temp)

    # Physics mode — can be overridden at CLI: physics_mode=rc
    physics_mode = str(cfg.get("physics_mode", "soc"))

    # KDE features for visualization (name → index, derived from DEFAULT_FEATURES)
    kde_features = {"SOC": soc_idx, "Voltage": volt_idx, "Current": curr_idx}

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------
    _stage("1 — Data Preparation (split → scale → sequence)")

    from src.data.dataset import process_and_save_data

    process_and_save_data(
        raw_data_path=cfg.paths.raw_data_file,
        processed_dir=processed_dir,
        scaler_save_path=scaler_path,
        test_size=0.2,
        random_state=seed,
        sequence_length=seq_len,
    )

    # Load the processed sequences produced by the pipeline
    X_train = np.load(os.path.join(processed_dir, "train_sequences.npy"))
    X_test  = np.load(os.path.join(processed_dir, "test_sequences.npy"))
    y_train = np.load(os.path.join(processed_dir, "train_conditioning.npy"))
    y_test  = np.load(os.path.join(processed_dir, "test_conditioning.npy"))

    num_features = X_train.shape[2]
    log.info(
        f"Data loaded — train: {X_train.shape}  test: {X_test.shape}  "
        f"features: {num_features}"
    )

    # ------------------------------------------------------------------
    # 2. Autoencoder training
    # ------------------------------------------------------------------
    _stage("2 — Autoencoder Training (BiLSTM encoder + LSTM decoder)")

    from src.models.variants import build_bilstm_autoencoder

    encoder, decoder, autoencoder = build_bilstm_autoencoder(
        seq_len=seq_len,
        num_features=num_features,
        latent_dim=latent_dim,
        lstm_units=lstm_units,
    )

    ae_epochs = int(cfg.training.get("ae_epochs", 50))
    log.info(f"Training autoencoder for {ae_epochs} epochs …")
    ae_history = autoencoder.fit(
        X_train, X_train,
        epochs=ae_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
    )

    # Persist history for training-curve plot
    import pandas as pd
    ae_history_df = pd.DataFrame(ae_history.history)
    ae_history_df.to_csv(os.path.join(metrics_dir, "ae_history.csv"), index=False)
    log.info(f"AE training history saved → {metrics_dir}/ae_history.csv")

    encoder.save(os.path.join(models_dir, "encoder.keras"))
    decoder.save(os.path.join(models_dir, "decoder.keras"))
    log.info(f"Autoencoder saved → {models_dir}/{{encoder,decoder}}.keras")

    # Extract real latent representations
    log.info("Extracting latent representations from training set …")
    train_latents = encoder.predict(X_train, batch_size=batch_size, verbose=0)
    np.save(os.path.join(processed_dir, "train_latents.npy"), train_latents)
    log.info(f"Latent vectors shape: {train_latents.shape}")

    # ------------------------------------------------------------------
    # 3. WGAN-GP training
    # ------------------------------------------------------------------
    _stage(f"3 — WGAN-GP Training  [physics_mode={physics_mode}]")

    from src.models.gan import build_generator, build_discriminator
    from src.training.train_gan import WGANTrainer

    generator     = build_generator(latent_dim=latent_dim,
                                    noise_dim=noise_dim,
                                    cond_dim=int(cfg.model.cond_dim))
    discriminator = build_discriminator(latent_dim=latent_dim,
                                        cond_dim=int(cfg.model.cond_dim))

    trainer = WGANTrainer(
        generator=generator,
        discriminator=discriminator,
        decoder=decoder,
        cfg=cfg,
        physics_mode=physics_mode,
    )
    trainer.fit(
        latent_reps=train_latents,
        conditions=y_train,
        save_dir=models_dir,
    )
    # trainer.fit() already saves models; re-bind the references from trainer
    generator     = trainer.generator
    discriminator = trainer.discriminator
    decoder       = trainer.decoder   # still frozen — will be thawed by trainer._save_models

    # ------------------------------------------------------------------
    # 4. Synthetic data generation
    # ------------------------------------------------------------------
    _stage("4 — Synthetic Data Generation")

    n_synthetic = int(cfg.get("n_synthetic", 5000))
    log.info(f"Generating {n_synthetic} synthetic sequences …")

    rng = np.random.default_rng(seed=seed)
    syn_noise = rng.standard_normal((n_synthetic, noise_dim)).astype(np.float32)
    # Sample conditioning values from the empirical training distribution
    syn_conds = y_train[rng.integers(0, len(y_train), size=n_synthetic)].astype(np.float32)

    syn_latents   = generator.predict([syn_noise, syn_conds],
                                      batch_size=batch_size, verbose=0)
    syn_sequences = decoder.predict(syn_latents, batch_size=batch_size, verbose=0)

    np.save(os.path.join(synthetic_dir, "synthetic_sequences.npy"), syn_sequences)
    np.save(os.path.join(synthetic_dir, "synthetic_latents.npy"),   syn_latents)
    np.save(os.path.join(synthetic_dir, "synthetic_conds.npy"),      syn_conds)
    log.info(
        f"Synthetic sequences saved → {synthetic_dir}  "
        f"shape: {syn_sequences.shape}"
    )

    # ------------------------------------------------------------------
    # 5. Evaluation — physics filter, MMD, plots
    # ------------------------------------------------------------------
    _stage("5 — Evaluation (Physics Filter → MMD → Visualization)")

    from src.evaluation.metrics import (
        evaluate_mmd, physics_filter, compute_all_metrics, save_metrics_json,
    )
    from src.evaluation.visualization import plot_all

    # Resolve evaluation config block (with defaults for backward compat)
    eval_cfg = cfg.get("evaluation", {})
    n_bins            = int(eval_cfg.get("n_bins",             50))
    autocorr_max_lag  = int(eval_cfg.get("autocorr_max_lag",   20))
    n_sequence_samples = int(eval_cfg.get("n_sequence_samples",  8))
    sequence_feature  = str(eval_cfg.get("sequence_feature",  "SOC"))

    # 5a. Physics rejection filter
    syn_filtered, conds_filtered = physics_filter(
        syn_sequences, syn_conds,
        soc_idx=soc_idx,
        volt_idx=volt_idx,
    )
    np.save(os.path.join(synthetic_dir, "synthetic_filtered.npy"),       syn_filtered)
    np.save(os.path.join(synthetic_dir, "synthetic_filtered_conds.npy"), conds_filtered)
    log.info(f"Filtered synthetic set: {syn_filtered.shape[0]} / {n_synthetic} retained")

    fake_seqs_eval = syn_filtered if len(syn_filtered) > 100 else syn_sequences

    # 5b. Encode test split for leakage-free latent metrics
    test_latents = encoder.predict(X_test, batch_size=batch_size, verbose=0)
    np.save(os.path.join(processed_dir, "test_latents.npy"), test_latents)

    # 5c. Full metric suite (test-split arrays used — no leakage)
    all_metrics = compute_all_metrics(
        real_sequences_test=X_test,
        fake_sequences=fake_seqs_eval,
        real_latents_train=train_latents,
        real_latents_test=test_latents,
        fake_latents=syn_latents,
        feature_indices=kde_features,
        soc_idx=soc_idx,
        volt_idx=volt_idx,
        n_bins=n_bins,
        autocorr_max_lag=autocorr_max_lag,
        random_state=seed,
    )
    mmd_metrics = all_metrics["mmd"]["train_split"]  # backward-compat reference
    log.info(
        f"MMD²(train)={mmd_metrics['mmd2']:.6f}  "
        f"MMD²(test)={all_metrics['mmd']['test_split']['mmd2']:.6f}  "
        f"γ={mmd_metrics['gamma']:.6f}"
    )

    # 5d. Diagnostic plots — all saved as .pdf
    import pandas as pd
    ae_history_path  = os.path.join(metrics_dir, "ae_history.csv")
    gan_history_path = os.path.join(metrics_dir, "gan_history.csv")
    ae_hist_df  = pd.read_csv(ae_history_path)  if os.path.isfile(ae_history_path)  else None
    gan_hist_df = pd.read_csv(gan_history_path) if os.path.isfile(gan_history_path) else None

    plot_all(
        real_sequences=X_test,
        fake_sequences=fake_seqs_eval,
        real_latents=test_latents,
        fake_latents=syn_latents,
        feature_indices=kde_features,
        figures_dir=figures_dir,
        random_state=seed,
        pass_rate_metrics=all_metrics.get("physics_pass_rate"),
        acf_curves=all_metrics.get("acf_curves"),
        ae_history_df=ae_hist_df,
        gan_history_df=gan_hist_df,
        sequence_sample_feature=sequence_feature,
        n_sequence_samples=n_sequence_samples,
    )

    # ------------------------------------------------------------------
    # 6. Downstream SOH prediction
    # ------------------------------------------------------------------
    _stage("6 — Downstream SOH Prediction")

    from src.evaluation.downstream_soh import run_downstream_soh

    soh_metrics = run_downstream_soh(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_syn=syn_filtered if len(syn_filtered) > 100 else syn_sequences,
        y_syn=conds_filtered if len(conds_filtered) > 100 else syn_conds,
        lstm_units=lstm_units,
        pretrain_epochs=int(cfg.training.get("soh_pretrain_epochs", 20)),
        finetune_epochs=int(cfg.training.get("soh_finetune_epochs", 10)),
        baseline_epochs=int(cfg.training.get("soh_baseline_epochs", 10)),
        batch_size=batch_size,
        seed=seed,
    )
    log.info(
        f"SOH  Baseline  → RMSE={soh_metrics['baseline']['rmse']:.6f}  "
        f"MAE={soh_metrics['baseline']['mae']:.6f}"
    )
    log.info(
        f"SOH  Pretrained → RMSE={soh_metrics['pretrained']['rmse']:.6f}  "
        f"MAE={soh_metrics['pretrained']['mae']:.6f}"
    )

    # Append SOH metrics to all_metrics and save the complete JSON
    all_metrics["soh"] = soh_metrics
    save_metrics_json(
        metrics=all_metrics,
        save_path=os.path.join(metrics_dir, "eval_metrics.json"),
    )

    # SOH comparison bar chart (now soh_metrics is available)
    from src.evaluation.visualization import plot_soh_comparison
    plot_soh_comparison(
        soh_metrics=soh_metrics,
        save_path=os.path.join(figures_dir, "soh_comparison.pdf"),
    )

    # ------------------------------------------------------------------
    # 7. Export results
    # ------------------------------------------------------------------
    _stage("7 — Exporting Results (CSV + LaTeX)")

    from src.utils.export import export_results

    export_results(
        soh_metrics=soh_metrics,
        mmd_metrics=mmd_metrics,
        metrics_dir=metrics_dir,
        tables_dir=tables_dir,
        caption=(
            "Downstream State-of-Health prediction (RMSE \\& MAE on real test set) "
            "and latent-space MMD$^2$ between real encoder representations and "
            "synthetic generator outputs."
        ),
        label="tab:main_results",
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - run_start
    log.info("")
    log.info("=" * 68)
    log.info(f"  PIPELINE COMPLETE  —  total time: {elapsed/60:.1f} min")
    log.info(f"  Config snapshot  : {config_snapshot_path}")
    log.info(f"  Metrics CSV      : {metrics_dir}/results.csv")
    log.info(f"  LaTeX table      : {tables_dir}/results.tex")
    log.info(f"  Figures          : {figures_dir}/")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
