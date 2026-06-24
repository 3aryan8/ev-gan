# ==============================================================================
# Standard LSTM Autoencoder
# ==============================================================================
# Architecture: Unidirectional LSTM encoder -> dense bottleneck -> RepeatVector
# decoder -> Unidirectional LSTM decoder -> per-timestep Dense output.
#
# Hydra config keys consumed:
#   model.latent_dim    — bottleneck dimensionality
#   model.lstm_units    — LSTM hidden units for both encoder and decoder
#   model.sequence_length — fixed sequence length (used to build decoder)
# ==============================================================================

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_encoder(
    seq_len: int,
    num_features: int,
    latent_dim: int,
    lstm_units: int = 64,
) -> Model:
    """
    Builds the standard unidirectional LSTM encoder.

    Args:
        seq_len:      Number of timesteps in each input sequence.
        num_features: Number of input features per timestep.
        latent_dim:   Dimensionality of the latent bottleneck vector.
        lstm_units:   Number of hidden units in the LSTM layer.

    Returns:
        Keras Model  Input: (batch, seq_len, num_features)
                     Output: (batch, latent_dim)
    """
    inputs = layers.Input(shape=(seq_len, num_features), name="encoder_input")
    x = layers.LSTM(lstm_units, return_sequences=False, name="encoder_lstm")(inputs)
    z = layers.Dense(latent_dim, activation="tanh", name="latent_vector")(x)
    return Model(inputs, z, name="encoder")


def build_decoder(
    seq_len: int,
    num_features: int,
    latent_dim: int,
    lstm_units: int = 64,
) -> Model:
    """
    Builds the standard unidirectional LSTM decoder.

    Args:
        seq_len:      Number of timesteps to reconstruct.
        num_features: Number of output features per timestep.
        latent_dim:   Dimensionality of the latent input vector.
        lstm_units:   Number of hidden units in the LSTM layer.

    Returns:
        Keras Model  Input: (batch, latent_dim)
                     Output: (batch, seq_len, num_features)
    """
    inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.RepeatVector(seq_len, name="repeat_vector")(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True, name="decoder_lstm")(x)
    outputs = layers.TimeDistributed(
        layers.Dense(num_features), name="decoder_output"
    )(x)
    return Model(inputs, outputs, name="decoder")


def build_autoencoder(
    seq_len: int,
    num_features: int,
    latent_dim: int,
    lstm_units: int = 64,
) -> Tuple[Model, Model, Model]:
    """
    Assembles the full LSTM Autoencoder from separate encoder and decoder sub-models
    and compiles it with MSE loss.

    Args:
        seq_len:      Number of timesteps in each input sequence.
        num_features: Number of input features per timestep.
        latent_dim:   Dimensionality of the latent bottleneck vector.
        lstm_units:   Number of hidden units shared across LSTM layers.

    Returns:
        Tuple of (encoder, decoder, autoencoder) Keras models.
    """
    encoder = build_encoder(seq_len, num_features, latent_dim, lstm_units)
    decoder = build_decoder(seq_len, num_features, latent_dim, lstm_units)

    ae_input = layers.Input(shape=(seq_len, num_features), name="autoencoder_input")
    ae_output = decoder(encoder(ae_input))
    autoencoder = Model(ae_input, ae_output, name="autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse"
    )

    return encoder, decoder, autoencoder
