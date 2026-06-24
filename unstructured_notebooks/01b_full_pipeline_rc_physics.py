import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration Constants
DATA_PATH = "/kaggle/input/evbattery-dataset-csv/all_battery_data_sampled.csv"
CLEANED_PATH = "cleaned_battery_data.csv"
SEGMENTED_PATH = "segmented_battery_data.csv"
NORMALIZED_PATH = "normalized_battery_data.csv"

FIXED_SEQ_LEN = 64
LATENT_DIM = 32
COND_DIM = 1
NOISE_DIM = 16
BATCH_SIZE = 64
EPOCHS_GAN = 2000
N_CRITIC = 5
LAMBDA_GP = 10
FEATURE_IDX_SOC = 6
FEATURE_IDX_VOLT = 0
FEATURE_IDX_CURRENT = 1

# ==========================================
# 1. DATA CLEANING & PREPROCESSING
# ==========================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans SOC and imputes zero capacities."""
    df['SOC'] = df['SOC'].clip(lower=0)
    
    zero_capacity_mask = df['capacity'] == 0
    mean_capacity = (
        df[~zero_capacity_mask]
        .groupby('Dataset')['capacity']
        .mean()
    )
    
    dataset_means = df['Dataset'].map(mean_capacity)
    df.loc[zero_capacity_mask, 'capacity'] = dataset_means[zero_capacity_mask]
    
    df.to_csv(CLEANED_PATH, index=False)
    return df

def segment_and_downsample(df: pd.DataFrame) -> pd.DataFrame:
    """Segments data by car/charge_segment and downsamples."""
    grouped = df.groupby(['car', 'charge_segment'])
    cycle_segments = []
    
    for (car_id, segment_id), group in grouped:
        group_sorted = group.sort_values('Timestamp')
        downsampled = group_sorted.iloc[::3].copy()  # 30s intervals from 10s
        downsampled['cycle_id'] = f"{car_id}_{segment_id}"
        cycle_segments.append(downsampled)
        
    df_segmented = pd.concat(cycle_segments, ignore_index=True)
    df_segmented.to_csv(SEGMENTED_PATH, index=False)
    return df_segmented

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies MinMax scaling to [-1, 1]."""
    columns_to_normalize = [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
        'mileage', 'capacity'
    ]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    df.to_csv(NORMALIZED_PATH, index=False)
    return df

# ==========================================
# 2. SEQUENCE FORMATTING & CONDITIONING
# ==========================================
def create_sequences(df: pd.DataFrame):
    """Converts dataframe into 3D padded sequences and extracts conditioning variables."""
    input_features = [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
        'mileage', 'capacity'
    ]
    
    grouped = df.groupby('cycle_id')
    sequences = []
    
    for _, group in grouped:
        group_sorted = group.sort_values('Timestamp')
        seq = group_sorted[input_features].to_numpy()
        sequences.append(seq)
        
    num_features = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), FIXED_SEQ_LEN, num_features), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len >= FIXED_SEQ_LEN:
            padded_sequences[i] = seq[:FIXED_SEQ_LEN]
        else:
            pad = np.tile(seq[-1], (FIXED_SEQ_LEN - seq_len, 1))
            padded_sequences[i] = np.vstack([seq, pad])
            
    np.save("battery_sequences_padded.npy", padded_sequences)
    
    conditioning_values = df.groupby("cycle_id")["capacity"].last().values.reshape(-1, 1)
    np.save("battery_conditioning.npy", conditioning_values)
    
    return padded_sequences, conditioning_values

# ==========================================
# 3. AUTOENCODER (LATENT SPACE MAPPING)
# ==========================================
def build_and_train_autoencoder(X: np.ndarray):
    """Builds, trains, and saves LSTM Autoencoder."""
    seq_len, num_features = X.shape[1], X.shape[2]
    
    encoder_inputs = layers.Input(shape=(seq_len, num_features))
    x = layers.LSTM(64, return_sequences=False)(encoder_inputs)
    z = layers.Dense(LATENT_DIM, activation="tanh")(x)
    encoder = Model(encoder_inputs, z, name="encoder")
    
    decoder_inputs = layers.Input(shape=(LATENT_DIM,))
    x = layers.RepeatVector(seq_len)(decoder_inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(num_features))(x)
    decoder = Model(decoder_inputs, decoded, name="decoder")
    
    autoencoder = Model(encoder_inputs, decoder(encoder(encoder_inputs)), name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    encoder.save("encoder_model.h5")
    decoder.save("decoder_model.h5")
    return encoder, decoder

# ==========================================
# 4. GAN ARCHITECTURE & RC PHYSICS INTEGRATION
# ==========================================
def build_generator() -> Model:
    noise_input = layers.Input(shape=(NOISE_DIM,))
    cond_input = layers.Input(shape=(COND_DIM,))
    
    x = layers.Concatenate()([noise_input, cond_input])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    latent_output = layers.Dense(LATENT_DIM, activation='tanh')(x)
    
    return Model([noise_input, cond_input], latent_output, name="Generator")

def build_discriminator() -> Model:
    latent_input = layers.Input(shape=(LATENT_DIM,))
    cond_input = layers.Input(shape=(COND_DIM,))

    x = layers.Concatenate()([latent_input, cond_input])
    x = layers.Dense(64, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    validity = layers.Dense(1)(x)  # No activation for WGAN-GP
    
    return Model([latent_input, cond_input], validity, name="Discriminator")

def build_rc_parameter_net(input_dim=3) -> Model:
    """Predicts RC circuit parameters (R0, R1, C1) based on conditioning."""
    cond_input = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu')(cond_input)
    x = layers.Dense(16, activation='relu')(x)
    rc_params = layers.Dense(3, activation='softplus')(x)  # Ensure R0, R1, C1 > 0
    return Model(cond_input, rc_params, name="RC_Param_Net")

def simulate_rc_voltage(current, rc_params, seq_len=FIXED_SEQ_LEN, dt=30.0):
    """Simulates voltage response using an RC circuit model."""
    rc_params = tf.maximum(rc_params, 1e-3)
    R0, R1, C1 = tf.split(rc_params, 3, axis=-1)
    
    # Clamp for stability
    R0 = tf.clip_by_value(R0, 0.01, 1.0)
    R1 = tf.clip_by_value(R1, 0.01, 2.0)
    C1 = tf.clip_by_value(C1, 50.0, 5000.0)

    batch_size = tf.shape(current)[0]
    V1 = tf.zeros([batch_size, 1], dtype=tf.float32)
    V_rc = []

    for t in range(seq_len):
        I_t = tf.reshape(current[:, t], [-1, 1])
        dV1 = (I_t - V1 / R1) / C1
        dV1 = tf.clip_by_value(dV1, -10.0, 10.0)
        V1 = V1 + dt * dV1
        V_t = I_t * R0 + V1
        V_rc.append(V_t)

    V_rc = tf.stack(V_rc, axis=1)
    return tf.squeeze(V_rc, axis=-1)

def gradient_penalty(real, fake, cond, discriminator):
    alpha = tf.random.uniform([real.shape[0], 1], 0., 1.)
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, cond])
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.) ** 2)

def train_gan_with_rc(X_real, y_cond, decoder):
    generator = build_generator()
    discriminator = build_discriminator()
    rc_param_net = build_rc_parameter_net(input_dim=3)
    
    g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    dataset = tf.data.Dataset.from_tensor_slices((X_real, y_cond)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_discriminator(real, cond):
        noise = tf.random.normal([real.shape[0], NOISE_DIM])
        fake = generator([noise, cond], training=True)
        with tf.GradientTape() as tape:
            d_real = discriminator([real, cond], training=True)
            d_fake = discriminator([fake, cond], training=True)
            gp = gradient_penalty(real, fake, cond, discriminator)
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + LAMBDA_GP * gp
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss

    @tf.function
    def train_generator_with_rc_physics(cond, lambda_rc=1.0, lambda_soc=5.0):
        noise = tf.random.normal([cond.shape[0], NOISE_DIM])
        with tf.GradientTape() as tape:
            fake_latent = generator([noise, cond], training=True)
            decoded = decoder(fake_latent, training=False)

            voltage = decoded[:, :, FEATURE_IDX_VOLT]
            current = decoded[:, :, FEATURE_IDX_CURRENT]
            soc     = decoded[:, :, FEATURE_IDX_SOC]

            avg_current = tf.reduce_mean(current, axis=1, keepdims=True)
            avg_soc = tf.reduce_mean(soc, axis=1, keepdims=True)
            rc_input = tf.concat([tf.cast(cond, tf.float32), avg_current, avg_soc], axis=1)
            rc_params = rc_param_net(rc_input)

            v_rc_sim = simulate_rc_voltage(current, rc_params)

            # Physics losses
            rc_loss = tf.reduce_mean(tf.square(voltage - v_rc_sim))
            soc_diff = soc[:, 1:] - soc[:, :-1]
            soc_violation = tf.nn.relu(-soc_diff)
            soc_loss = tf.reduce_mean(soc_violation)

            # Adversarial loss
            d_fake = discriminator([fake_latent, cond], training=True)
            adv_loss = -tf.reduce_mean(d_fake)

            # Total Generator Loss
            lambda_adv = 2.0
            g_loss = lambda_adv * adv_loss + lambda_rc * rc_loss + lambda_soc * soc_loss

        grads = tape.gradient(g_loss, generator.trainable_variables + rc_param_net.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables + rc_param_net.trainable_variables))
        return g_loss, adv_loss, rc_loss, soc_loss

    for epoch in range(EPOCHS_GAN):
        for step, (real_batch, cond_batch) in enumerate(dataset):
            for _ in range(N_CRITIC):
                d_loss = train_discriminator(real_batch, cond_batch)
            
            # RC decay schedule
            lambda_rc = 1.0 * (0.95 ** (epoch / 1000.0))
            g_loss, adv_loss, rc_loss, soc_loss = train_generator_with_rc_physics(cond_batch, lambda_rc=lambda_rc)
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss = {d_loss.numpy():.4f}, G_loss = {g_loss.numpy():.4f}, RC = {rc_loss.numpy():.4f}, SOC = {soc_loss.numpy():.4f}")

    generator.save("gan_generator.h5")
    discriminator.save("gan_discriminator.h5")
    rc_param_net.save("rc_param_net.h5")
    return generator, discriminator, rc_param_net

# ==========================================
# 5. EVALUATION & GENERATION
# ==========================================
def evaluate_and_generate(generator, decoder, X_real, y_cond):
    """Runs KDE, t-SNE, and generates filtered synthetic dataset."""
    n_samples = 10000
    noise = np.random.normal(size=(n_samples, NOISE_DIM))
    sampled_cond = y_cond[np.random.choice(len(y_cond), n_samples, replace=True)]
    
    latent_fake = generator.predict([noise, sampled_cond], verbose=0)
    synthetic_sequences = decoder.predict(latent_fake, verbose=0)
    
    # KDE Plot
    features = {"SOC": 6, "Average Voltage": 0, "Charging Current": 1}
    for name, idx in features.items():
        real_values = X_real[:, :, idx].flatten()
        fake_values = synthetic_sequences[:, :, idx].flatten()
        plt.figure(figsize=(10, 4))
        sns.kdeplot(real_values, label="Real", fill=True, color='blue')
        sns.kdeplot(fake_values, label="Synthetic", fill=True, color='orange')
        plt.title(f"KDE Comparison for {name}")
        plt.legend()
        plt.close()
        
    # t-SNE Plot
    latent_real = np.load("latent_vectors.npy")
    n_tsne = min(1000, len(latent_real))
    noise_tsne = np.random.normal(size=(n_tsne, NOISE_DIM))
    cond_tsne = y_cond[np.random.choice(len(y_cond), n_tsne, replace=False)]
    fake_tsne = generator.predict([noise_tsne, cond_tsne], verbose=0)
    
    X_combined = np.concatenate([latent_real[:n_tsne], fake_tsne], axis=0)
    labels = ['Real'] * n_tsne + ['Synthetic'] * n_tsne
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X_combined)
    
    # Physics Filter
    def passes_physics_filter(seq):
        soc = seq[:, FEATURE_IDX_SOC]
        volt = seq[:, FEATURE_IDX_VOLT]
        if np.sum(np.diff(soc) < -0.01) > 2: return False
        if np.var(soc) < 0.0005: return False
        if volt[-1] < volt[0] - 0.01: return False
        return True

    mask = np.array([passes_physics_filter(seq) for seq in synthetic_sequences])
    X_syn_filtered = synthetic_sequences[mask]
    y_syn_filtered = sampled_cond[mask]
    
    np.save("synthetic_filtered.npy", X_syn_filtered)
    np.save("synthetic_filtered_capacities.npy", y_syn_filtered)
    return X_syn_filtered, y_syn_filtered

# ==========================================
# 6. DOWNSTREAM SOH PREDICTION
# ==========================================
def run_downstream_soh(X_real, y_real, X_syn, y_syn):
    """Trains baseline vs synthetic-pretrained LSTM for SOH prediction."""
    y_real = y_real.flatten()
    y_syn = y_syn.flatten()
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    def build_lstm():
        model = Sequential([
            LSTM(64, input_shape=(FIXED_SEQ_LEN, 10), return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    model_pretrained = build_lstm()
    model_pretrained.fit(X_syn, y_syn, epochs=20, batch_size=32, verbose=0)
    model_pretrained.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)

    model_real = build_lstm()
    model_real.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)

    def evaluate(model, X, y, label):
        pred = model.predict(X, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        print(f"{label} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    print("\nBaseline: Real-Only Model on Real Test Set:")
    evaluate(model_real, X_test, y_test, "Real-only model")
    print("\nFinal Evaluation on Real Test Set:")
    evaluate(model_pretrained, X_test, y_test, "Pretrained on Synthetic -> Fine-tuned on Real")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Data Prep
    df_raw = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_seg = segment_and_downsample(df_clean)
    df_norm = normalize_data(df_seg)
    X_pad, y_cond = create_sequences(df_norm)
    
    # 2. Autoencoder
    encoder, decoder = build_and_train_autoencoder(X_pad)
    latent_vectors = encoder.predict(X_pad, batch_size=32)
    np.save("latent_vectors.npy", latent_vectors)
    
    # 3. GAN Training with RC Physics
    generator, discriminator, rc_net = train_gan_with_rc(latent_vectors, y_cond, decoder)
    
    # 4. Evaluation & Generation
    X_syn, y_syn = evaluate_and_generate(generator, decoder, X_pad, y_cond)
    
    # 5. Downstream Task
    run_downstream_soh(X_pad, y_cond, X_syn, y_syn)
    
    print(f"Total Execution Time: {time.time() - start_time:.2f}s")