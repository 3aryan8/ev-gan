"""
Enterprise Deep Learning Pipeline: 
Physics-Informed Latent WGAN-GP for EV Battery Time-Series

Architecture: Raw CSV -> Temporal Interpolation -> BiLSTM Autoencoder -> Latent WGAN-GP -> Downstream SOH Regressor
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

# Configure Enterprise Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("PhysGAN-Pipeline")


@dataclass
class PipelineConfig:
    """Master Configuration object to eliminate hardcoded magic numbers."""
    data_path: str = "./all_battery_data_sampled.csv"
    output_dir: str = "./artifacts/"
    
    # Dimensions
    seq_len: int = 64
    num_features: int = 10
    latent_dim: int = 32
    noise_dim: int = 16
    
    # Training Hyperparameters
    batch_size: int = 64
    ae_epochs: int = 50
    gan_epochs: int = 1500
    n_critic: int = 5
    lambda_gp: float = 10.0
    lambda_physics: float = 10.0
    test_split_ratio: float = 0.2
    random_seed: int = 42

    feature_cols: List[str] = field(default_factory=lambda: [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp', 'mileage', 'capacity'
    ])
    
    def __post_init__(self):
        self.soc_idx = self.feature_cols.index('SOC')
        os.makedirs(self.output_dir, exist_ok=True)


# =====================================================================
# 1. DATA PREPROCESSING & QUARANTINE
# =====================================================================

class BatteryDataPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def _interpolate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Replaces dumb static padding with physically sound linear time-warping."""
        current_len = sequence.shape[0]
        if current_len == self.cfg.seq_len:
            return sequence
            
        old_steps = np.linspace(0, 1, current_len)
        new_steps = np.linspace(0, 1, self.cfg.seq_len)
        
        resampled = np.zeros((self.cfg.seq_len, self.cfg.num_features))
        for i in range(self.cfg.num_features):
            resampled[:, i] = np.interp(new_steps, old_steps, sequence[:, i])
        return resampled

    def ingest_and_quarantine(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Executes safe group-splitting before scaling to prevent data leakage."""
        logger.info("Starting raw data ingestion...")
        df = pd.read_csv(self.cfg.data_path)
        
        # Domain Clean 1: Clip SOC
        df['SOC'] = df['SOC'].clip(lower=0)
        
        # Domain Clean 2: Impute missing capacities group-wise
        mean_cap = df[df['capacity'] > 0].groupby('Dataset')['capacity'].mean()
        df['capacity'] = df.apply(lambda r: mean_cap.get(r['Dataset'], 0) if r['capacity'] == 0 else r['capacity'], axis=1)

        # Create unique cycle tracking keys
        df['cycle_id'] = df['car'].astype(str) + "_" + df['charge_segment'].astype(str)

        # Domain Clean 3: Fix thermal sensor glitch
        swap_mask = df["Min_Cell_Temperature"] > df["Max_Cell_Temperature"]
        df.loc[swap_mask, ["Max_Cell_Temperature", "Min_Cell_Temperature"]] = (
            df.loc[swap_mask, ["Min_Cell_Temperature", "Max_Cell_Temperature"]].values
        )

        # QUARANTINE STEP: Group split by vehicle_session so whole cycles stay intact
        splitter = GroupShuffleSplit(test_size=self.cfg.test_split_ratio, n_splits=1, random_state=self.cfg.random_seed)
        train_idx, test_idx = next(splitter.split(df, groups=df['cycle_id']))
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Fit scaler ONLY on training data
        train_df[self.cfg.feature_cols] = self.scaler.fit_transform(train_df[self.cfg.feature_cols])
        test_df[self.cfg.feature_cols] = self.scaler.transform(test_df[self.cfg.feature_cols])

        train_data = self._pack_sequences(train_df)
        test_data = self._pack_sequences(test_df)

        logger.info(f"Quarantine successful. Train sequences: {train_data['X'].shape[0]} | Test: {test_data['X'].shape[0]}")
        return train_data, test_data

    def _pack_sequences(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        sequences, conditions = [], []
        for _, group in df.groupby('cycle_id'):
            seq = group.sort_values('Timestamp')[self.cfg.feature_cols].values[::3] # Downsample interval 3
            if len(seq) < 10: 
                continue # drop micro-glitch sessions
                
            resampled_seq = self._interpolate_sequence(seq)
            sequences.append(resampled_seq)
            conditions.append(group['capacity'].iloc[-1]) # Target conditioning SOH

        return {
            "X": np.array(sequences, dtype=np.float32),
            "cond": np.array(conditions, dtype=np.float32).reshape(-1, 1)
        }


# =====================================================================
# 2. DIMENSIONAL AIRLOCK (BiLSTM AUTOENCODER)
# =====================================================================

class LatentAutoencoder:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.encoder, self.decoder, self.autoencoder = self._build_subnets()

    def _build_subnets(self):
        # Encoder
        enc_in = layers.Input(shape=(self.cfg.seq_len, self.cfg.num_features))
        x = layers.Bidirectional(layers.LSTM(64))(enc_in)
        latent_out = layers.Dense(self.cfg.latent_dim, activation="tanh")(x)
        encoder = Model(enc_in, latent_out, name="BiLSTM_Encoder")

        # Decoder
        dec_in = layers.Input(shape=(self.cfg.latent_dim,))
        x = layers.RepeatVector(self.cfg.seq_len)(dec_in)
        x = layers.LSTM(64, return_sequences=True)(x)
        dec_out = layers.TimeDistributed(layers.Dense(self.cfg.num_features))(x)
        decoder = Model(dec_in, dec_out, name="BiLSTM_Decoder")

        # Composite AE
        autoencoder = Model(enc_in, decoder(encoder(enc_in)), name="BiLSTM_Autoencoder")
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
        return encoder, decoder, autoencoder

    def train_and_compress(self, X_train: np.ndarray) -> np.ndarray:
        logger.info("Training Latent Autoencoder...")
        self.autoencoder.fit(
            X_train, X_train, 
            epochs=self.cfg.ae_epochs, 
            batch_size=self.cfg.batch_size, 
            validation_split=0.1, 
            verbose=0
        )
        self.encoder.save(os.path.join(self.cfg.output_dir, "encoder.keras"))
        self.decoder.save(os.path.join(self.cfg.output_dir, "decoder.keras"))
        
        logger.info("Extracting Latent representations...")
        return self.encoder.predict(X_train, batch_size=self.cfg.batch_size, verbose=0)


# =====================================================================
# 3. THERMODYNAMIC WGAN-GP 
# =====================================================================

class PhysicsInformedWGANTrainer:
    def __init__(self, cfg: PipelineConfig, frozen_decoder: Model):
        self.cfg = cfg
        self.decoder = frozen_decoder
        self.decoder.trainable = False 
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        self.g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    def _build_generator(self) -> Model:
        noise_in = layers.Input(shape=(self.cfg.noise_dim,))
        cond_in = layers.Input(shape=(1,))
        x = layers.Concatenate()([noise_in, cond_in])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(self.cfg.latent_dim, activation='tanh')(x)
        return Model([noise_in, cond_in], out, name="Generator")

    def _build_discriminator(self) -> Model:
        latent_in = layers.Input(shape=(self.cfg.latent_dim,))
        cond_in = layers.Input(shape=(1,))
        x = layers.Concatenate()([latent_in, cond_in])
        x = layers.Dense(64, activation='leaky_relu')(x)
        x = layers.Dense(64, activation='leaky_relu')(x)
        out = layers.Dense(1)(x) # Linear output for WGAN
        return Model([latent_in, cond_in], out, name="Discriminator")

    @tf.function
    def _gradient_penalty(self, real_lat, fake_lat, cond):
        alpha = tf.random.uniform([real_lat.shape[0], 1], 0., 1.)
        interpolated = alpha * real_lat + (1. - alpha) * fake_lat
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator([interpolated, cond], training=True)
        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        return tf.reduce_mean((norm - 1.) ** 2)

    @tf.function
    def _train_d_step(self, real_latent, cond):
        noise = tf.random.normal([real_latent.shape[0], self.cfg.noise_dim])
        with tf.GradientTape() as d_tape:
            fake_latent = self.generator([noise, cond], training=True)
            d_real = self.discriminator([real_latent, cond], training=True)
            d_fake = self.discriminator([fake_latent, cond], training=True)
            
            gp = self._gradient_penalty(real_latent, fake_latent, cond)
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + (self.cfg.lambda_gp * gp)
            
        grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return d_loss

    @tf.function
    def _train_g_step(self, cond):
        batch_s = tf.shape(cond)[0]
        noise = tf.random.normal([batch_s, self.cfg.noise_dim])
        
        with tf.GradientTape() as g_tape:
            fake_latent = self.generator([noise, cond], training=True)
            d_fake = self.discriminator([fake_latent, cond], training=True)
            adv_loss = -tf.reduce_mean(d_fake)

            # --- SOFT PHYSICS LOSS (The Conservation of Charge penalty) ---
            decoded_seq = self.decoder(fake_latent, training=False)
            soc_curve = decoded_seq[:, :, self.cfg.soc_idx]
            soc_diffs = soc_curve[:, 1:] - soc_curve[:, :-1]
            # If diff is negative, battery "lost" charge. Penalize.
            physics_violation = tf.nn.relu(-soc_diffs)
            physics_penalty = tf.reduce_mean(physics_violation)

            total_g_loss = adv_loss + (self.cfg.lambda_physics * physics_penalty)

        grads = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return total_g_loss, physics_penalty

    def fit(self, latent_reps: np.ndarray, conditions: np.ndarray):
        logger.info("Training Physics-Informed WGAN-GP...")
        dataset = tf.data.Dataset.from_tensor_slices((latent_reps, conditions)).shuffle(1024).batch(self.cfg.batch_size)
        
        for epoch in range(1, self.cfg.gan_epochs + 1):
            for real_b, cond_b in dataset:
                for _ in range(self.cfg.n_critic):
                    self._train_d_step(real_b, cond_b)
                g_loss, phys_loss = self._train_g_step(cond_b)
                
            if epoch % 300 == 0 or epoch == self.cfg.gan_epochs:
                logger.info(f"GAN Epoch {epoch}/{self.cfg.gan_epochs} | G_Loss: {g_loss:.4f} | Phys_Violation: {phys_loss:.6f}")

        self.generator.save(os.path.join(self.cfg.output_dir, "generator.keras"))


# =====================================================================
# 4. REJECTION SANITIZER & VALIDATION
# =====================================================================

class ProductionSanitizer:
    @staticmethod
    def reject_bad_physics(sequences: np.ndarray, capacities: np.ndarray, soc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Hard rejection sampling of synthetic data."""
        valid_idx = []
        for i, seq in enumerate(sequences):
            soc = seq[:, soc_idx]
            volt = seq[:, 0]
            
            # Rule 1: No more than 2 minor negative SOC ticks
            if np.sum(np.diff(soc) < -0.01) > 2:
                continue
            # Rule 2: Cannot have a totally flatlined SOC curve
            if np.var(soc) < 0.0005:
                continue
            # Rule 3: End voltage cannot be substantially lower than start voltage during a charge
            if volt[-1] < (volt[0] - 0.01):
                continue
            valid_idx.append(i)
            
        logger.info(f"Sanitizer passed {len(valid_idx)} / {len(sequences)} synthetic samples.")
        return sequences[valid_idx], capacities[valid_idx]


def evaluate_downstream_utility(X_real_train, y_real_train, X_real_test, y_real_test, X_syn, y_syn):
    """Proves the ROI of the pipeline via a downstream State-of-Health Regressor."""
    logger.info("Executing Sim-to-Real downstream proof...")
    
    def get_regressor():
        model = Sequential([
            layers.LSTM(64, input_shape=(64, 10)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # Baseline: Trained strictly on real data
    logger.info("Training Baseline Model (Real Data Only)...")
    m_baseline = get_regressor()
    m_baseline.fit(X_real_train, y_real_train, epochs=12, batch_size=32, verbose=0)
    base_preds = m_baseline.predict(X_real_test, verbose=0)
    
    # Pre-trained: Warmed up on Synthetic, fine-tuned on Real
    logger.info("Training Upgraded Model (Pretrained on Syn -> Fine-tuned on Real)...")
    m_upgraded = get_regressor()
    m_upgraded.fit(X_syn, y_syn, epochs=15, batch_size=32, verbose=0) # Pretrain
    m_upgraded.fit(X_real_train, y_real_train, epochs=6, batch_size=32, verbose=0) # Fine-tune
    upg_preds = m_upgraded.predict(X_real_test, verbose=0)

    logger.info("=== DOWNSTREAM SOH PREDICTION RESULTS ===")
    logger.info(f"Baseline (Real only) -> RMSE: {np.sqrt(mean_squared_error(y_real_test, base_preds)):.4f} | MAE: {mean_absolute_error(y_real_test, base_preds):.4f}")
    logger.info(f"PhysGAN Upgraded   -> RMSE: {np.sqrt(mean_squared_error(y_real_test, upg_preds)):.4f} | MAE: {mean_absolute_error(y_real_test, upg_preds):.4f}")


# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================

if __name__ == "__main__":
    start_t = time.time()
    cfg = PipelineConfig()
    
    # 1. Ingest & Quarantine
    pipeline = BatteryDataPipeline(cfg)
    train_data, test_data = pipeline.ingest_and_quarantine()
    
    # 2. Compress to Latent Space
    ae = LatentAutoencoder(cfg)
    train_latent = ae.train_and_compress(train_data["X"])
    
    # 3. Train GAN
    gan = PhysicsInformedWGANTrainer(cfg, ae.decoder)
    gan.fit(train_latent, train_data["cond"])
    
    # 4. Generate 5,000 raw synthetic cycles
    logger.info("Generating 5,000 synthetic candidates...")
    syn_noise = np.random.normal(size=(5000, cfg.noise_dim))
    # Sample random conditioning capacities from the known training distribution
    syn_conds = np.random.choice(train_data["cond"].flatten(), size=(5000, 1))
    
    raw_syn_latent = gan.generator.predict([syn_noise, syn_conds], batch_size=128, verbose=0)
    raw_syn_sequences = ae.decoder.predict(raw_syn_latent, batch_size=128, verbose=0)
    
    # 5. Sanitize synthetic data
    clean_syn_seq, clean_syn_cond = ProductionSanitizer.reject_bad_physics(
        raw_syn_sequences, syn_conds, cfg.soc_idx
    )
    
    # 6. Prove ROI
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_data["X"], train_data["cond"], test_size=0.2, random_state=42
    )
    
    evaluate_downstream_utility(
        X_real_train=X_train_split, y_real_train=y_train_split,
        X_real_test=test_data["X"], y_real_test=test_data["cond"],
        X_syn=clean_syn_seq, y_syn=clean_syn_cond
    )
    
    logger.info(f"Pipeline executed successfully in {(time.time() - start_t)/60:.2f} minutes.")