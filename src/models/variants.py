# ==============================================================================
# Autoencoder Variants
# ==============================================================================
# Contains richer encoder architectures that drop into the same decoder as the
# standard autoencoder.  All builders share the same call signature as their
# counterparts in autoencoder.py so they are interchangeable.
#
# Variants implemented:
#   1. BiLSTM Autoencoder  — Bidirectional LSTM encoder (most common in notebooks)
#
# Hydra config keys consumed (same as autoencoder.py):
#   model.latent_dim  |  model.lstm_units  |  model.sequence_length
# ==============================================================================

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model

from src.models.autoencoder import build_decoder  # shared decoder is reused as-is


def build_bilstm_encoder(
    seq_len: int,
    num_features: int,
    latent_dim: int,
    lstm_units: int = 64,
) -> Model:
    """
    Bidirectional LSTM encoder.

    The Bidirectional wrapper doubles the effective hidden state width, so the
    LSTM layer internally has ``lstm_units`` units in each direction, producing a
    concatenated output of size ``2 * lstm_units`` that is then projected to
    ``latent_dim`` via a dense tanh layer.

    Args:
        seq_len:      Number of timesteps in each input sequence.
        num_features: Number of input features per timestep.
        latent_dim:   Dimensionality of the latent bottleneck vector.
        lstm_units:   Number of LSTM units per direction.

    Returns:
        Keras Model  Input: (batch, seq_len, num_features)
                     Output: (batch, latent_dim)
    """
    inputs = layers.Input(shape=(seq_len, num_features), name="bilstm_encoder_input")
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm_encoder_lstm",
    )(inputs)
    z = layers.Dense(latent_dim, activation="tanh", name="latent_vector")(x)
    return Model(inputs, z, name="bilstm_encoder")


def build_bilstm_autoencoder(
    seq_len: int,
    num_features: int,
    latent_dim: int,
    lstm_units: int = 64,
) -> Tuple[Model, Model, Model]:
    """
    Assembles the BiLSTM Autoencoder: Bidirectional encoder + standard decoder.

    This is the architecture used in 01d, 01g, 01h, 01i, 01k, 01l, and the
    physgan_ev_pipeline notebooks.

    Args:
        seq_len:      Number of timesteps in each input sequence.
        num_features: Number of input features per timestep.
        latent_dim:   Dimensionality of the latent bottleneck vector.
        lstm_units:   Number of LSTM units (per direction for encoder).

    Returns:
        Tuple of (encoder, decoder, autoencoder) Keras models.
    """
    encoder = build_bilstm_encoder(seq_len, num_features, latent_dim, lstm_units)
    decoder = build_decoder(seq_len, num_features, latent_dim, lstm_units)

    ae_input = layers.Input(shape=(seq_len, num_features), name="bilstm_autoencoder_input")
    ae_output = decoder(encoder(ae_input))
    autoencoder = Model(ae_input, ae_output, name="bilstm_autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse"
    )

    return encoder, decoder, autoencoder
