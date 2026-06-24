import os
import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration Constants
DATA_PATH = "/kaggle/input/evbattery-dataset-csv/all_battery_data_sampled.csv"
CLEANED_PATH = "cleaned_battery_data.csv"
SEGMENTED_PATH = "segmented_battery_data.csv"
NORMALIZED_PATH = "normalized_battery_data.csv"
SCALER_PATH = "battery_scaler.pkl"

FIXED_SEQ_LEN = 64
LATENT_DIM = 32
COND_DIM = 1
NOISE_DIM = 16
BATCH_SIZE = 64
EPOCHS_GAN = 2000
N_CRITIC = 5
LAMBDA_GP = 10.0
LAMBDA_PHYS = 1.0
FEATURE_IDX_SOC = 6
FEATURE_IDX_VOLT = 0
FEATURE_IDX_CURR = 1
FEATURE_IDX_TEMP = 4

# ==========================================
# 1. DATA CLEANING & PREPROCESSING
# ==========================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['SOC'] = df['SOC'].clip(lower=0)
    zero_capacity_mask = df['capacity'] == 0
    mean_capacity = df[~zero_capacity_mask].groupby('Dataset')['capacity'].mean()
    dataset_means = df['Dataset'].map(mean_capacity)
    df.loc[zero_capacity_mask, 'capacity'] = dataset_means[zero_capacity_mask]
    df.to_csv(CLEANED_PATH, index=False)
    return df

def segment_and_downsample(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(['car', 'charge_segment'])
    cycle_segments = []
    for (car_id, segment_id), group in grouped:
        group_sorted = group.sort_values('Timestamp')
        downsampled = group_sorted.iloc[::3].copy()
        downsampled['cycle_id'] = f"{car_id}_{segment_id}"
        cycle_segments.append(downsampled)
    df_segmented = pd.concat(cycle_segments, ignore_index=True)
    df_segmented.to_csv(SEGMENTED_PATH, index=False)
    return df_segmented

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_normalize = [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
        'mileage', 'capacity'
    ]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    joblib.dump(scaler, SCALER_PATH)
    df.to_csv(NORMALIZED_PATH, index=False)
    return df

# ==========================================
# 2. SEQUENCE FORMATTING & CONDITIONING
# ==========================================
def create_sequences(df: pd.DataFrame):
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
# 3. AUTOENCODER
# ==========================================
def build_and_train_autoencoder(X: np.ndarray):
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
# 4. GAN ARCHITECTURE & ADVANCED PHYSICS LOSS
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
    validity = layers.Dense(1)(x)
    return Model([latent_input, cond_input], validity, name="Discriminator")

def compute_physics_losses(decoded, cond):
    """Calculates SOC monotonicity, energy consistency, and temperature smoothness."""
    decoded = tf.cast(decoded, tf.float32)
    cond = tf.cast(cond, tf.float32)
    eps = 1e-6

    soc = decoded[:, :, FEATURE_IDX_SOC]
    curr = decoded[:, :, FEATURE_IDX_CURR]
    temp = decoded[:, :, FEATURE_IDX_TEMP]

    # Ensure positive, bounded "capacity" scale
    capacity = tf.clip_by_value((cond + 2.0) / 2.0, 0.25, 2.0)

    # 1. SOC monotonic increase
    soc_diff = soc[:, 1:] - soc[:, :-1]
    soc_loss = tf.reduce_mean(tf.nn.relu(-soc_diff))

    # 2. Energy consistency
    cum_curr = tf.cumsum(curr, axis=1)
    delta_soc = soc - soc[:, :1]
    predicted_delta = cum_curr / (capacity + eps)
    energy_loss = tf.reduce_mean(tf.square(delta_soc - predicted_delta)) * 1e-6

    # 3. Smooth temperature change
    dtemp = temp[:, 1:] - temp[:, :-1]
    temp_loss = tf.reduce_mean(tf.square(dtemp)) * 0.01

    phys_total = soc_loss + energy_loss + temp_loss
    return phys_total, soc_loss, energy_loss, temp_loss

def gradient_penalty(real, fake, cond, discriminator):
    alpha = tf.random.uniform([real.shape[0], 1], 0., 1.)
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator([interpolated, cond])
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.) ** 2)

def train_gan(X_real, y_cond_raw, decoder):
    # Normalize conditioning variables to [-1, 1]
    cap_min, cap_max = y_cond_raw.min(), y_cond_raw.max()
    y_cond = 2 * (y_cond_raw - cap_min) / (cap_max - cap_min) - 1
    
    generator = build_generator()
    discriminator = build_discriminator()
    
    g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    dataset = tf.data.Dataset.from_tensor_slices((X_real.astype(np.float32), y_cond.astype(np.float32))).shuffle(10000).batch(BATCH_SIZE).prefetch(1)

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
    def train_generator_with_physics(cond):
        noise = tf.random.normal([cond.shape[0], NOISE_DIM])
        with tf.GradientTape() as tape:
            fake_latent = generator([noise, cond], training=True)
            decoded = decoder(fake_latent, training=False)
            
            phys_total, soc_loss, energy_loss, temp_loss = compute_physics_losses(decoded, cond)
            
            d_fake = discriminator([fake_latent, cond], training=True)
            adv_loss = -tf.reduce_mean(d_fake)
            g_loss = adv_loss + LAMBDA_PHYS * phys_total
            
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss, adv_loss, phys_total, soc_loss, energy_loss, temp_loss

    for epoch in range(EPOCHS_GAN):
        for step, (real_batch, cond_batch) in enumerate(dataset):
            for _ in range(N_CRITIC):
                d_loss = train_discriminator(real_batch, cond_batch)
            g_loss, adv_loss, phys_total, soc_loss, energy_loss, temp_loss = train_generator_with_physics(cond_batch)
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_adv={adv_loss:.4f}, Phys={phys_total:.4f} (SOC={soc_loss:.4f}, E={energy_loss:.4f}, T={temp_loss:.4f})")

    generator.save("gan_generator.h5")
    discriminator.save("gan_discriminator.h5")
    return generator, discriminator

# ==========================================
# 5. EVALUATION, GENERATION & MMD
# ==========================================
def compute_mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

def evaluate_and_generate(generator, decoder, X_real, y_cond):
    n_samples = 10000
    noise = np.random.normal(size=(n_samples, NOISE_DIM))
    sampled_cond = y_cond[np.random.choice(len(y_cond), n_samples, replace=True)]
    
    latent_fake = generator.predict([noise, sampled_cond], verbose=0)
    synthetic_sequences = decoder.predict(latent_fake, verbose=0)
    np.save("synthetic_battery_cycles.npy", synthetic_sequences)
    np.save("synthetic_capacities.npy", sampled_cond)
    
    # MMD Calculation
    latent_real = np.load("latent_vectors.npy")
    n_mmd = min(1000, len(latent_real))
    cond_mmd = y_cond[np.random.choice(len(y_cond), n_mmd, replace=False)]
    noise_mmd = np.random.normal(size=(n_mmd, NOISE_DIM))
    fake_mmd = generator.predict([noise_mmd, cond_mmd], verbose=0)
    
    print("\n📊 MMD² between Real and Synthetic Latents:")
    for gamma in [0.1, 0.5, 1.0, 2.0]:
        mmd = compute_mmd(latent_real[:n_mmd], fake_mmd, gamma=gamma)
        print(f"γ = {gamma:<4} → MMD² = {mmd:.6f}")

    # KDE Plot
    features = {"SOC": 6, "Average Voltage": 0, "Charging Current": 1}
    for name, idx in features.items():
        real_values = X_real[:, :, idx].flatten()
        fake_values = synthetic_sequences[:, :, idx].flatten()
        plt.figure(figsize=(8, 3))
        sns.kdeplot(real_values, label="Real", fill=True)
        sns.kdeplot(fake_values, label="Synthetic", fill=True, color='orange')
        plt.title(f"KDE: {name}")
        plt.close()
        
    # t-SNE Plot
    n_tsne = min(1000, len(latent_real))
    noise_tsne = np.random.normal(size=(n_tsne, NOISE_DIM))
    cond_tsne = y_cond[np.random.choice(len(y_cond), n_tsne, replace=False)]
    fake_tsne = generator.predict([noise_tsne, cond_tsne], verbose=0)
    
    X_combined = np.concatenate([latent_real[:n_tsne], fake_tsne], axis=0)
    labels = ['Real'] * n_tsne + ['Synthetic'] * n_tsne
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X_combined)
    
    # Denormalize Synthetic Data
    scaler = joblib.load(SCALER_PATH)
    X_denorm_flat = scaler.inverse_transform(synthetic_sequences.reshape(-1, 10))
    X_denorm = X_denorm_flat.reshape(synthetic_sequences.shape)
    
    # Convert Denormalized to CSV
    feature_names = [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp', 'mileage', 'capacity'
    ]
    rows = []
    for seq_id in range(X_denorm.shape[0]):
        for t in range(X_denorm.shape[1]):
            row = {"sequence_id": seq_id, "timestep": t, "conditioned_capacity": sampled_cond[seq_id, 0]}
            row.update({feature_names[i]: X_denorm[seq_id, t, i] for i in range(len(feature_names))})
            rows.append(row)
    pd.DataFrame(rows).to_csv("synthetic_battery_data_denormalized.csv", index=False)
    
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
    total_start_time = time.time()
    
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
    
    # 3. GAN Training with Advanced Physics
    generator, discriminator = train_gan(latent_vectors, y_cond, decoder)
    
    # 4. Evaluation, Generation & Denormalization
    X_syn, y_syn = evaluate_and_generate(generator, decoder, X_pad, y_cond)
    
    # 5. Downstream Task
    run_downstream_soh(X_pad, y_cond, X_syn, y_syn)
    
    print(f"Total Execution Time: {time.time() - total_start_time:.2f}s")