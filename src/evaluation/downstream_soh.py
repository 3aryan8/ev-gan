# ==============================================================================
# Downstream SOH (State-of-Health) Prediction Task
# ==============================================================================
# Extracted and improved from all notebook variants.
#
# Bug fixed vs. notebooks:
#   The original code performed a SECOND train_test_split inside this function
#   on the full real dataset, meaning the test fold it used for LSTM evaluation
#   did not correspond to the preprocessing-level split.  This module accepts
#   explicit (X_train, X_test, y_train, y_test) arguments that are produced
#   by the upstream preprocessing pipeline — the same split used for scaling.
#
# Two model configs are evaluated:
#   "baseline"   — LSTM trained only on real training data.
#   "pretrained" — LSTM pre-trained on synthetic data, fine-tuned on real.
#
# All metrics are returned as a dict (never only printed) so they can be
# consumed by src/utils/export.py for CSV/LaTeX export.
# ==============================================================================

import logging
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, Sequential

log = logging.getLogger(__name__)


def _build_soh_lstm(seq_len: int, num_features: int, lstm_units: int = 64) -> Sequential:
    """
    Builds the LSTM regressor for SOH (capacity) prediction.

    Architecture:
        LSTM(lstm_units) → Dense(32, relu) → Dense(1, linear)

    Args:
        seq_len:      Number of timesteps per sequence.
        num_features: Number of features per timestep.
        lstm_units:   Hidden units in the LSTM layer. Default 64.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential(
        [
            layers.Input(shape=(seq_len, num_features)),
            layers.LSTM(lstm_units, return_sequences=False, name="lstm"),
            layers.Dense(32, activation="relu", name="dense_hidden"),
            layers.Dense(1, name="soh_output"),
        ],
        name="soh_regressor",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Returns RMSE and MAE as unrounded floats."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def run_downstream_soh(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    lstm_units: int = 64,
    pretrain_epochs: int = 20,
    finetune_epochs: int = 10,
    baseline_epochs: int = 10,
    val_split: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Trains and evaluates two LSTM regressors for State-of-Health (SOH/capacity)
    prediction, comparing a real-data-only baseline against a synthetic
    pre-trained + fine-tuned model.

    Both models are evaluated **only** on the held-out real test split —
    the same split produced by the upstream preprocessing pipeline.

    Args:
        X_train:          Real training sequences (N_train, seq_len, num_features).
        y_train:          Real training capacity targets (N_train,) or (N_train, 1).
        X_test:           Real test sequences (N_test, seq_len, num_features).
        y_test:           Real test capacity targets (N_test,) or (N_test, 1).
        X_syn:            Synthetic sequences for pre-training (N_syn, seq_len, num_features).
        y_syn:            Synthetic capacity targets (N_syn,) or (N_syn, 1).
        lstm_units:       Hidden units in both LSTM regressors. Default 64.
        pretrain_epochs:  Epochs for synthetic pre-training phase. Default 20.
        finetune_epochs:  Epochs for real fine-tuning phase. Default 10.
        baseline_epochs:  Epochs for the real-only baseline. Default 10.
        val_split:        Fraction of training data used for validation. Default 0.1.
        batch_size:       Mini-batch size. Default 32.
        seed:             Random seed for TF weight initialisation. Default 42.

    Returns:
        Dict with keys ``"baseline"`` and ``"pretrained"``, each containing
        ``{"rmse": float, "mae": float}`` evaluated on the real test split.

    Example::

        results = run_downstream_soh(
            X_train, y_train, X_test, y_test, X_syn, y_syn
        )
        # results == {
        #   "baseline":   {"rmse": 0.0421, "mae": 0.0318},
        #   "pretrained": {"rmse": 0.0387, "mae": 0.0291},
        # }
    """
    tf.random.set_seed(seed)

    y_train = y_train.flatten().astype(np.float32)
    y_test  = y_test.flatten().astype(np.float32)
    y_syn   = y_syn.flatten().astype(np.float32)

    seq_len, num_features = X_train.shape[1], X_train.shape[2]

    # ------------------------------------------------------------------
    # Model A: Baseline — trained only on real data
    # ------------------------------------------------------------------
    log.info("[SOH] Training baseline model (real data only) …")
    model_baseline = _build_soh_lstm(seq_len, num_features, lstm_units)
    model_baseline.fit(
        X_train, y_train,
        epochs=baseline_epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=0,
    )
    pred_baseline = model_baseline.predict(X_test, verbose=0).flatten()
    metrics_baseline = _compute_metrics(y_test, pred_baseline)
    log.info(
        f"[SOH] Baseline  → RMSE={metrics_baseline['rmse']:.6f}  "
        f"MAE={metrics_baseline['mae']:.6f}"
    )

    # ------------------------------------------------------------------
    # Model B: Pretrained on synthetic, fine-tuned on real
    # ------------------------------------------------------------------
    log.info("[SOH] Pre-training on synthetic data …")
    model_pretrained = _build_soh_lstm(seq_len, num_features, lstm_units)
    model_pretrained.fit(
        X_syn, y_syn,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        verbose=0,
    )

    log.info("[SOH] Fine-tuning on real training data …")
    model_pretrained.fit(
        X_train, y_train,
        epochs=finetune_epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=0,
    )
    pred_pretrained = model_pretrained.predict(X_test, verbose=0).flatten()
    metrics_pretrained = _compute_metrics(y_test, pred_pretrained)
    log.info(
        f"[SOH] Pretrained → RMSE={metrics_pretrained['rmse']:.6f}  "
        f"MAE={metrics_pretrained['mae']:.6f}"
    )

    return {
        "baseline":   metrics_baseline,
        "pretrained": metrics_pretrained,
    }
