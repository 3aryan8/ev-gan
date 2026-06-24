# ==============================================================================
# RC Circuit Model
# ==============================================================================
# Extracted from 01b_full_pipeline_rc_physics.py.
#
# The first-order Thevenin equivalent circuit (RC model) models a battery cell
# as a series resistance R0 and a parallel RC branch (R1 || C1):
#
#   V_terminal = I * R0 + V1
#   dV1/dt     = (I - V1/R1) / C1
#
# At each discrete timestep (dt = 30 s after downsampling):
#   V1[t] = V1[t-1] + dt * clip(dV1, -10, 10)
#   V[t]  = I[t] * R0 + V1[t]
#
# The RC parameter network predicts (R0, R1, C1) conditioned on the per-cycle
# mean current, mean SOC, and capacity so that the simulation adapts to the
# specific charge profile.
#
# Notebooks of origin: 01b_full_pipeline_rc_physics.py
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_rc_parameter_net(input_dim: int = 3) -> Model:
    """
    Predicts RC circuit parameters (R0, R1, C1) from a per-cycle feature vector.

    The input feature vector is expected to contain (at minimum):
        [capacity, mean_current, mean_soc]
    though its dimensionality is controlled by ``input_dim`` for flexibility.

    Softplus output activations enforce strictly positive parameter values,
    which is a physical requirement of the circuit model.

    Args:
        input_dim: Dimensionality of the conditioning input vector.
                   Default 3 = [capacity, mean_current, mean_soc].

    Returns:
        Keras Model  Input:  (batch, input_dim)
                     Output: (batch, 3) → [R0, R1, C1]
    """
    cond_input = tf.keras.Input(shape=(input_dim,), name="rc_cond_input")
    x = layers.Dense(16, activation="relu", name="rc_dense_1")(cond_input)
    x = layers.Dense(16, activation="relu", name="rc_dense_2")(x)
    # softplus → output is always > 0; physically mandatory for R and C
    rc_params = layers.Dense(3, activation="softplus", name="rc_params")(x)
    return Model(cond_input, rc_params, name="rc_parameter_net")


@tf.function
def simulate_rc_voltage(
    current: tf.Tensor,
    rc_params: tf.Tensor,
    seq_len: int,
    dt: float = 30.0,
) -> tf.Tensor:
    """
    Simulates the terminal voltage of a first-order RC (Thevenin) battery model
    over ``seq_len`` timesteps using the Euler forward integration method.

    Parameter clamping prevents numerical blow-up during training when the
    RC parameter network is still learning:
        R0  ∈ [0.01, 1.0]   Ω
        R1  ∈ [0.01, 2.0]   Ω
        C1  ∈ [50.0, 5000.0] F

    Args:
        current:   Per-timestep charging current, shape (batch, seq_len).
                   Feature slice from the decoded sequence.
        rc_params: Predicted RC parameters, shape (batch, 3) → [R0, R1, C1].
                   Produced by ``build_rc_parameter_net``.
        seq_len:   Number of timesteps to simulate (must match current dim 1).
        dt:        Integration timestep in seconds. Default 30 s (post-downsampling).

    Returns:
        Simulated terminal voltage, shape (batch, seq_len).
    """
    rc_params = tf.maximum(rc_params, 1e-3)                # global floor
    R0, R1, C1 = tf.split(rc_params, 3, axis=-1)          # each (batch, 1)

    # Per-parameter physical clamps
    R0 = tf.clip_by_value(R0, 0.01, 1.0)
    R1 = tf.clip_by_value(R1, 0.01, 2.0)
    C1 = tf.clip_by_value(C1, 50.0, 5000.0)

    batch_size = tf.shape(current)[0]
    V1 = tf.zeros([batch_size, 1], dtype=tf.float32)  # RC branch voltage, init 0
    V_rc = []

    for t in range(seq_len):
        I_t = tf.reshape(current[:, t], [-1, 1])      # (batch, 1)
        # Euler step: dV1/dt = (I - V1/R1) / C1
        dV1 = (I_t - V1 / R1) / C1
        dV1 = tf.clip_by_value(dV1, -10.0, 10.0)      # gradient stability clamp
        V1 = V1 + dt * dV1
        V_t = I_t * R0 + V1                           # terminal voltage
        V_rc.append(V_t)

    # Stack into (batch, seq_len, 1) then squeeze last dim → (batch, seq_len)
    return tf.squeeze(tf.stack(V_rc, axis=1), axis=-1)


def build_rc_input(
    decoded: tf.Tensor,
    cond: tf.Tensor,
    curr_idx: int,
    soc_idx: int,
) -> tf.Tensor:
    """
    Constructs the 3-dimensional input vector for the RC parameter network from
    a decoded sequence tensor and its conditioning capacity.

    The vector is ``[mean_capacity, mean_current, mean_soc]``, matching the
    input convention used in 01b_full_pipeline_rc_physics.py.

    Args:
        decoded:   Decoded sequence tensor, shape (batch, seq_len, num_features).
        cond:      Capacity conditioning tensor, shape (batch, 1).
        curr_idx:  Column index of Charging_Current in the feature dimension.
        soc_idx:   Column index of SOC in the feature dimension.

    Returns:
        RC input tensor, shape (batch, 3).
    """
    decoded = tf.cast(decoded, tf.float32)
    cond = tf.cast(cond, tf.float32)

    avg_current = tf.reduce_mean(decoded[:, :, curr_idx], axis=1, keepdims=True)
    avg_soc = tf.reduce_mean(decoded[:, :, soc_idx], axis=1, keepdims=True)
    return tf.concat([cond, avg_current, avg_soc], axis=1)  # (batch, 3)
