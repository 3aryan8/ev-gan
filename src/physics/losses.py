# ==============================================================================
# Physics-Informed Loss Functions
# ==============================================================================
# Three distinct physics losses extracted from the exploratory notebooks.
# All functions accept decoded sequence tensors and feature indices (derived
# from the feature name list), never hardcoded integers.
#
# Feature reference (DEFAULT_FEATURES in src/data/sequences.py):
#   index 0  – Average_Cell_Voltage
#   index 1  – Charging_Current
#   index 2  – Max_Cell_Voltage
#   index 3  – Min_Cell_Voltage
#   index 4  – Max_Cell_Temperature
#   index 5  – Min_Cell_Temperature
#   index 6  – SOC
#   index 7  – Timestamp
#   index 8  – mileage
#   index 9  – capacity
#
# Notebooks of origin:
#   soc_monotonicity_loss  ← 01_full_pipeline_prototype.py / 01g / 01h / 01k / 01l
#   rc_circuit_loss        ← 01b_full_pipeline_rc_physics.py
#   advanced_physics_loss  ← 01l_full_pipeline_energy_physics.py
# ==============================================================================

import tensorflow as tf


@tf.function
def soc_monotonicity_loss(
    decoded: tf.Tensor,
    soc_idx: int,
) -> tf.Tensor:
    """
    Penalises any decrease in State-of-Charge between consecutive timesteps.

    During a charging cycle the SOC should never decrease.  Any negative
    first-difference is a physics violation and is penalised with a ReLU,
    so only drops contribute to the loss (increases are ignored).

    Args:
        decoded:  Decoded sequence tensor, shape (batch, seq_len, num_features).
        soc_idx:  Column index of the SOC feature in the last dimension.
                  Derive with ``feature_list.index('SOC')``.

    Returns:
        Scalar loss tensor.
    """
    decoded = tf.cast(decoded, tf.float32)
    soc = decoded[:, :, soc_idx]                     # (batch, seq_len)
    soc_diff = soc[:, 1:] - soc[:, :-1]             # (batch, seq_len - 1)
    violation = tf.nn.relu(-soc_diff)                # Only penalise decreases
    return tf.reduce_mean(violation)


@tf.function
def rc_circuit_loss(
    decoded: tf.Tensor,
    v_rc_simulated: tf.Tensor,
    volt_idx: int,
) -> tf.Tensor:
    """
    Computes the mean-squared error between the decoded voltage profile and
    the voltage profile predicted by the RC-circuit simulation.

    The RC simulation itself lives in ``src/physics/rc_model.py``.  This
    function only measures the discrepancy between model output and simulation,
    making it composable with other losses inside the training loop.

    Args:
        decoded:        Decoded sequence tensor, shape (batch, seq_len, num_features).
        v_rc_simulated: Simulated RC voltage, shape (batch, seq_len).
                        Produced by ``src.physics.rc_model.simulate_rc_voltage``.
        volt_idx:       Column index of the voltage feature.
                        Derive with ``feature_list.index('Average_Cell_Voltage')``.

    Returns:
        Scalar loss tensor.
    """
    decoded = tf.cast(decoded, tf.float32)
    v_rc_simulated = tf.cast(v_rc_simulated, tf.float32)
    voltage = decoded[:, :, volt_idx]               # (batch, seq_len)
    return tf.reduce_mean(tf.square(voltage - v_rc_simulated))


@tf.function
def advanced_physics_loss(
    decoded: tf.Tensor,
    cond: tf.Tensor,
    soc_idx: int,
    curr_idx: int,
    temp_idx: int,
    energy_scale: float = 1e-6,
    temp_scale: float = 0.01,
) -> tuple:
    """
    Compound physics loss combining three terms from 01l_full_pipeline_energy_physics.py:

    1. **SOC monotonicity** – identical to ``soc_monotonicity_loss`` above.
    2. **Energy consistency** – the cumulative current integral should track
       the change in SOC, scaled by the conditioning capacity.  A 1e-6 scale
       factor is applied because the raw units are much larger than the SOC
       scale after normalisation.
    3. **Temperature smoothness** – the squared first-difference of temperature
       is penalised to encourage physically plausible, smooth thermal curves.

    Args:
        decoded:       Decoded sequence tensor, shape (batch, seq_len, num_features).
        cond:          Conditioning capacity tensor, shape (batch, 1).
                       Expected to be in the scaled ``[-1, 1]`` range.
        soc_idx:       Column index of SOC.
        curr_idx:      Column index of Charging_Current.
        temp_idx:      Column index of Max_Cell_Temperature.
        energy_scale:  Multiplicative weight for the energy consistency term.
                       Default ``1e-6`` (matches the notebook).
        temp_scale:    Multiplicative weight for the temperature smoothness term.
                       Default ``0.01`` (matches the notebook).

    Returns:
        Tuple of (total_loss, soc_loss, energy_loss, temp_loss) scalar tensors.
    """
    decoded = tf.cast(decoded, tf.float32)
    cond = tf.cast(cond, tf.float32)
    eps = 1e-6

    soc  = decoded[:, :, soc_idx]       # (batch, seq_len)
    curr = decoded[:, :, curr_idx]      # (batch, seq_len)
    temp = decoded[:, :, temp_idx]      # (batch, seq_len)

    # Re-map conditioning from [-1, 1] to a physically bounded capacity scale.
    # The notebook uses (cond + 2) / 2 clamped to [0.25, 2.0].
    capacity = tf.clip_by_value((cond + 2.0) / 2.0, 0.25, 2.0)  # (batch, 1)

    # 1. SOC monotonic increase
    soc_diff = soc[:, 1:] - soc[:, :-1]
    soc_loss = tf.reduce_mean(tf.nn.relu(-soc_diff))

    # 2. Energy consistency: cumulative current should track ΔSOC / capacity
    cum_curr   = tf.cumsum(curr, axis=1)                 # (batch, seq_len)
    delta_soc  = soc - soc[:, :1]                        # (batch, seq_len)
    predicted_delta = cum_curr / (capacity + eps)        # (batch, seq_len)
    energy_loss = tf.reduce_mean(
        tf.square(delta_soc - predicted_delta)
    ) * energy_scale

    # 3. Temperature smoothness: penalise large step-to-step jumps
    dtemp = temp[:, 1:] - temp[:, :-1]
    temp_loss = tf.reduce_mean(tf.square(dtemp)) * temp_scale

    total = soc_loss + energy_loss + temp_loss
    return total, soc_loss, energy_loss, temp_loss
