# ==============================================================================
# WGAN-GP Generator & Discriminator
# ==============================================================================
# Latent-space conditional GAN.  The Generator maps (noise, capacity) -> latent
# vector; the Discriminator scores (latent, capacity) pairs.  A frozen decoder
# is used inside the generator's training step (in training.py) to decode and
# apply the physics penalty.
#
# Hydra config keys consumed:
#   model.latent_dim   — bottleneck / generator output dimensionality
#   model.noise_dim    — generator noise input dimensionality
#   model.cond_dim     — conditioning variable dimensionality (default 1 = capacity)
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_generator(
    latent_dim: int,
    noise_dim: int,
    cond_dim: int = 1,
) -> Model:
    """
    Conditional Generator: maps (noise, conditioning) -> latent vector.

    The generator is a simple MLP that concatenates noise and conditioning
    inputs, passes them through two hidden layers, and outputs a latent vector
    with ``tanh`` activation (matching the ``[-1, 1]`` scaled latent space
    produced by the autoencoder encoder).

    Args:
        latent_dim: Dimensionality of the output latent vector.
        noise_dim:  Dimensionality of the random noise input.
        cond_dim:   Dimensionality of the conditioning input. Default 1 (capacity).

    Returns:
        Keras Model  Inputs:  [(batch, noise_dim), (batch, cond_dim)]
                     Output:  (batch, latent_dim)
    """
    noise_input = layers.Input(shape=(noise_dim,), name="noise_input")
    cond_input = layers.Input(shape=(cond_dim,), name="cond_input")

    x = layers.Concatenate(name="concat_noise_cond")([noise_input, cond_input])
    x = layers.Dense(64, activation="relu", name="gen_dense_1")(x)
    x = layers.Dense(64, activation="relu", name="gen_dense_2")(x)
    latent_output = layers.Dense(latent_dim, activation="tanh", name="gen_output")(x)

    return Model([noise_input, cond_input], latent_output, name="generator")


def build_discriminator(
    latent_dim: int,
    cond_dim: int = 1,
) -> Model:
    """
    Conditional Discriminator (Critic): scores (latent, conditioning) pairs.

    Linear (no sigmoid) output as required by WGAN-GP.  LeakyReLU activations
    help avoid dead neurons in the critic with sparse gradients.

    Args:
        latent_dim: Dimensionality of the latent input.
        cond_dim:   Dimensionality of the conditioning input. Default 1 (capacity).

    Returns:
        Keras Model  Inputs:  [(batch, latent_dim), (batch, cond_dim)]
                     Output:  (batch, 1)  — raw critic score, no activation
    """
    latent_input = layers.Input(shape=(latent_dim,), name="latent_input")
    cond_input = layers.Input(shape=(cond_dim,), name="cond_input")

    x = layers.Concatenate(name="concat_latent_cond")([latent_input, cond_input])
    x = layers.Dense(64, name="disc_dense_1")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_lrelu_1")(x)
    x = layers.Dense(64, name="disc_dense_2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_lrelu_2")(x)
    validity = layers.Dense(1, name="disc_output")(x)  # Linear — no activation

    return Model([latent_input, cond_input], validity, name="discriminator")


@tf.function
def gradient_penalty(
    real: tf.Tensor,
    fake: tf.Tensor,
    cond: tf.Tensor,
    discriminator: Model,
) -> tf.Tensor:
    """
    Computes the WGAN-GP gradient penalty on interpolated samples.

    The penalty enforces the 1-Lipschitz constraint on the critic by penalising
    the squared deviation of the gradient norm from 1.

    Args:
        real:          Real latent vectors, shape (batch, latent_dim).
        fake:          Generated latent vectors, shape (batch, latent_dim).
        cond:          Conditioning vectors, shape (batch, cond_dim).
        discriminator: The critic model.

    Returns:
        Scalar penalty tensor.
    """
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
    interpolated = alpha * real + (1.0 - alpha) * fake

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, cond], training=True)

    grads = tape.gradient(pred, interpolated)
    # Euclidean norm across the latent dimension, small epsilon for numerical stability
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)
