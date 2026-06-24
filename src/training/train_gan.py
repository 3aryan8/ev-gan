# ==============================================================================
# WGAN-GP Training Loop
# ==============================================================================
# Extracts and unifies the training logic from:
#   01_full_pipeline_prototype.py  — baseline SOC-only physics step
#   01b_full_pipeline_rc_physics.py — RC-circuit physics step
#   01l_full_pipeline_energy_physics.py — advanced energy+temperature step
#   physgan_ev_pipeline.py        — OOP PhysicsInformedWGANTrainer reference
#
# The WGANTrainer class is configured entirely from the Hydra OmegaConf object
# (or any object with matching attributes).  No magic numbers live inside.
#
# Physics mode is selected via the ``physics_mode`` constructor argument:
#   "soc"      — SOC monotonicity only (default, fast, most notebooks)
#   "advanced" — energy consistency + temperature smoothness + SOC
#   "rc"       — RC-circuit voltage matching + SOC (requires rc_param_net)
# ==============================================================================

import os
import logging
import time
from typing import Optional

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras import Model

from src.physics.losses import (
    soc_monotonicity_loss,
    rc_circuit_loss,
    advanced_physics_loss,
)
from src.physics.rc_model import (
    build_rc_parameter_net,
    simulate_rc_voltage,
    build_rc_input,
)
from src.models.gan import gradient_penalty

log = logging.getLogger(__name__)


class WGANTrainer:
    """
    Physics-Informed WGAN-GP Trainer.

    Manages the full training lifecycle:
      - N-critic discriminator steps with gradient penalty per generator step
      - Physics-informed generator step (mode-selectable)
      - ``tf.data.Dataset`` pipeline with shuffle + prefetch
      - Periodic console logging
      - Model persistence to ``artifacts/models/``

    Args:
        generator:     Keras Generator model from ``src.models.gan.build_generator``.
        discriminator: Keras Discriminator (critic) model.
        decoder:       Trained AE decoder model (frozen during GAN training).
        cfg:           Hydra ``DictConfig`` with keys under ``training``,
                       ``model``, and ``feature_indices``.
        physics_mode:  One of ``"soc"`` | ``"advanced"`` | ``"rc"``.
                       Default ``"soc"``.

    Example::

        trainer = WGANTrainer(gen, disc, decoder, cfg, physics_mode="advanced")
        trainer.fit(train_latents, train_conds)
    """

    VALID_MODES = ("soc", "advanced", "rc")

    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        decoder: Model,
        cfg: DictConfig,
        physics_mode: str = "soc",
    ) -> None:
        if physics_mode not in self.VALID_MODES:
            raise ValueError(
                f"physics_mode must be one of {self.VALID_MODES}, got '{physics_mode}'"
            )

        self.generator = generator
        self.discriminator = discriminator
        self.cfg = cfg
        self.physics_mode = physics_mode

        # Freeze the decoder — it is only used for physics evaluation,
        # never updated during GAN training.
        self.decoder = decoder
        self.decoder.trainable = False

        # Feature indices — derived from config, never hardcoded
        fi = cfg.feature_indices
        self.soc_idx  = int(fi.soc)
        self.volt_idx = int(fi.volt)
        self.curr_idx = int(fi.curr)
        self.temp_idx = int(fi.temp)

        # Training hyperparameters
        tr = cfg.training
        self.n_critic       = int(tr.n_critic)
        self.lambda_gp      = float(tr.lambda_gp)
        self.lambda_physics = float(tr.lambda_physics)
        self.lambda_rc      = float(tr.lambda_rc)
        self.lambda_soc     = float(tr.lambda_soc)
        self.epochs_gan     = int(tr.epochs_gan)
        self.batch_size     = int(tr.batch_size)
        self.shuffle_buffer = int(tr.shuffle_buffer)
        self.log_every      = int(tr.log_every)
        self.noise_dim      = int(cfg.model.noise_dim)

        # Optimisers — Adam with WGAN-GP recommended β values
        self.g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

        # RC parameter network (only instantiated when needed)
        self.rc_param_net: Optional[Model] = None
        if self.physics_mode == "rc":
            self.rc_param_net = build_rc_parameter_net(input_dim=3)
            log.info("[WGANTrainer] RC parameter network initialised.")

    # ------------------------------------------------------------------
    # Discriminator step
    # ------------------------------------------------------------------

    @tf.function
    def _train_d_step(
        self,
        real_latent: tf.Tensor,
        cond: tf.Tensor,
    ) -> tf.Tensor:
        """
        Single discriminator (critic) update step.

        Computes Wasserstein loss + gradient penalty, then applies gradients to
        the discriminator.  Generator weights are NOT updated here.

        Args:
            real_latent: Real encoder latent vectors (batch, latent_dim).
            cond:        Conditioning capacity vectors (batch, cond_dim).

        Returns:
            Discriminator loss scalar.
        """
        batch_size = tf.shape(real_latent)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_latent = self.generator([noise, cond], training=True)

            d_real = self.discriminator([real_latent, cond], training=True)
            d_fake = self.discriminator([fake_latent, cond], training=True)

            gp = gradient_penalty(real_latent, fake_latent, cond, self.discriminator)

            # Wasserstein critic loss + GP regularisation
            d_loss = (
                tf.reduce_mean(d_fake)
                - tf.reduce_mean(d_real)
                + self.lambda_gp * gp
            )

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        return d_loss

    # ------------------------------------------------------------------
    # Generator steps (one per physics mode)
    # ------------------------------------------------------------------

    @tf.function
    def _train_g_step_soc(self, cond: tf.Tensor) -> tuple:
        """
        Generator step with SOC monotonicity physics loss only.

        This is the mode used in most notebooks (01g, 01h, 01i, 01k, 01l baseline).

        Returns:
            (g_loss, adv_loss, phys_loss) scalars.
        """
        batch_size = tf.shape(cond)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_latent = self.generator([noise, cond], training=True)
            decoded = self.decoder(fake_latent, training=False)

            phys_loss = soc_monotonicity_loss(decoded, self.soc_idx)

            d_fake = self.discriminator([fake_latent, cond], training=True)
            adv_loss = -tf.reduce_mean(d_fake)

            g_loss = adv_loss + self.lambda_physics * phys_loss

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )
        return g_loss, adv_loss, phys_loss

    @tf.function
    def _train_g_step_advanced(self, cond: tf.Tensor) -> tuple:
        """
        Generator step with the advanced compound physics loss from
        01l_full_pipeline_energy_physics.py:
            SOC monotonicity + energy consistency + temperature smoothness.

        Returns:
            (g_loss, adv_loss, phys_total, soc_loss, energy_loss, temp_loss) scalars.
        """
        batch_size = tf.shape(cond)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_latent = self.generator([noise, cond], training=True)
            decoded = self.decoder(fake_latent, training=False)

            phys_total, soc_loss, energy_loss, temp_loss = advanced_physics_loss(
                decoded, cond,
                soc_idx=self.soc_idx,
                curr_idx=self.curr_idx,
                temp_idx=self.temp_idx,
            )

            d_fake = self.discriminator([fake_latent, cond], training=True)
            adv_loss = -tf.reduce_mean(d_fake)

            g_loss = adv_loss + self.lambda_physics * phys_total

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )
        return g_loss, adv_loss, phys_total, soc_loss, energy_loss, temp_loss

    @tf.function
    def _train_g_step_rc(self, cond: tf.Tensor) -> tuple:
        """
        Generator step with RC-circuit physics loss from
        01b_full_pipeline_rc_physics.py.
        Trains BOTH the generator and the RC parameter network jointly.

        Returns:
            (g_loss, adv_loss, rc_loss, soc_loss) scalars.
        """
        batch_size = tf.shape(cond)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        trainable_vars = (
            self.generator.trainable_variables
            + self.rc_param_net.trainable_variables
        )

        with tf.GradientTape() as tape:
            fake_latent = self.generator([noise, cond], training=True)
            decoded = self.decoder(fake_latent, training=False)

            # Build per-batch RC conditioning vector from decoded sequence
            rc_input = build_rc_input(decoded, cond, self.curr_idx, self.soc_idx)
            rc_params = self.rc_param_net(rc_input, training=True)

            # Simulate RC voltage and compare to decoded voltage
            current = decoded[:, :, self.curr_idx]
            seq_len = tf.shape(decoded)[1]
            v_rc_sim = simulate_rc_voltage(current, rc_params, seq_len=seq_len)

            rc_loss  = rc_circuit_loss(decoded, v_rc_sim, self.volt_idx)
            soc_loss = soc_monotonicity_loss(decoded, self.soc_idx)

            d_fake = self.discriminator([fake_latent, cond], training=True)
            adv_loss = -tf.reduce_mean(d_fake)

            g_loss = (
                2.0 * adv_loss
                + self.lambda_rc  * rc_loss
                + self.lambda_soc * soc_loss
            )

        grads = tape.gradient(g_loss, trainable_vars)
        self.g_opt.apply_gradients(zip(grads, trainable_vars))
        return g_loss, adv_loss, rc_loss, soc_loss

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def fit(
        self,
        latent_reps: np.ndarray,
        conditions: np.ndarray,
        save_dir: str = "artifacts/models",
    ) -> None:
        """
        Runs the full WGAN-GP training loop.

        Builds a ``tf.data.Dataset`` from the provided latent representations
        and conditioning arrays, then runs ``epochs_gan`` epochs with
        ``n_critic`` discriminator steps per generator step.

        After training, saves the generator, discriminator, and decoder to
        ``save_dir`` in the native Keras ``.keras`` format.

        Args:
            latent_reps: Encoder latent vectors, shape (N, latent_dim).
            conditions:  Capacity conditioning, shape (N, cond_dim).
            save_dir:    Directory for saved models. Created if absent.
        """
        # Build prefetched tf.data pipeline
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                (
                    latent_reps.astype(np.float32),
                    conditions.astype(np.float32),
                )
            )
            .shuffle(self.shuffle_buffer)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        log.info(
            f"[WGANTrainer] Starting training | mode={self.physics_mode} "
            f"epochs={self.epochs_gan} n_critic={self.n_critic}"
        )
        t_start = time.time()

        for epoch in range(1, self.epochs_gan + 1):
            d_loss_last = tf.constant(0.0)
            g_metrics: tuple = ()

            for real_batch, cond_batch in dataset:
                # --- Critic steps ---
                for _ in range(self.n_critic):
                    d_loss_last = self._train_d_step(real_batch, cond_batch)

                # --- Generator step (mode-dispatched) ---
                if self.physics_mode == "soc":
                    g_metrics = self._train_g_step_soc(cond_batch)
                elif self.physics_mode == "advanced":
                    g_metrics = self._train_g_step_advanced(cond_batch)
                elif self.physics_mode == "rc":
                    g_metrics = self._train_g_step_rc(cond_batch)

            # Periodic logging (last batch metrics)
            if epoch % self.log_every == 0 or epoch == self.epochs_gan:
                self._log_epoch(epoch, d_loss_last, g_metrics)

        elapsed = time.time() - t_start
        log.info(f"[WGANTrainer] Training complete in {elapsed:.1f}s")

        self._save_models(save_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_epoch(self, epoch: int, d_loss: tf.Tensor, g_metrics: tuple) -> None:
        base = f"Epoch {epoch:>5}/{self.epochs_gan} | D={d_loss.numpy():.4f}"
        if self.physics_mode == "soc":
            g, adv, phys = [m.numpy() for m in g_metrics]
            log.info(f"{base} | G={g:.4f} adv={adv:.4f} soc={phys:.4f}")
        elif self.physics_mode == "advanced":
            g, adv, pt, sl, el, tl = [m.numpy() for m in g_metrics]
            log.info(
                f"{base} | G={g:.4f} adv={adv:.4f} "
                f"phys={pt:.4f} (soc={sl:.4f} E={el:.4f} T={tl:.4f})"
            )
        elif self.physics_mode == "rc":
            g, adv, rc, soc = [m.numpy() for m in g_metrics]
            log.info(
                f"{base} | G={g:.4f} adv={adv:.4f} rc={rc:.4f} soc={soc:.4f}"
            )

    def _save_models(self, save_dir: str) -> None:
        """Persists generator, discriminator, and decoder to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)

        gen_path  = os.path.join(save_dir, "generator.keras")
        disc_path = os.path.join(save_dir, "discriminator.keras")
        dec_path  = os.path.join(save_dir, "decoder.keras")

        self.generator.save(gen_path)
        self.discriminator.save(disc_path)
        # Re-enable decoder weights before saving so it can be loaded cleanly
        self.decoder.trainable = True
        self.decoder.save(dec_path)
        self.decoder.trainable = False  # restore freeze state

        if self.physics_mode == "rc" and self.rc_param_net is not None:
            rc_path = os.path.join(save_dir, "rc_param_net.keras")
            self.rc_param_net.save(rc_path)
            log.info(f"[WGANTrainer] RC param net saved → {rc_path}")

        log.info(
            f"[WGANTrainer] Models saved:\n"
            f"  Generator    → {gen_path}\n"
            f"  Discriminator → {disc_path}\n"
            f"  Decoder      → {dec_path}"
        )
