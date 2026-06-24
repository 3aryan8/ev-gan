# Physics-Informed Battery GAN (PIC-GAN)

A research-grade, modular Python repository for Physics-Informed Generative Adversarial Networks applied to EV battery data.

This repository refactors monolithic Jupyter notebook experiments into a scalable pipeline powered by `Hydra`.

## Repository Structure

```
pic_gan_battery/
├── configs/                  # Hydra/YAML configuration files
│   └── config.yaml           # Main configuration (seeds, paths, training hyperparameters)
├── src/                      # Core source code
│   ├── data/                 # Data preprocessing and sequence generation
│   ├── evaluation/           # MMD, KDE/t-SNE/PCA visualization, SOH downstream task
│   ├── models/               # Autoencoder, BiLSTM, and WGAN-GP definitions
│   ├── physics/              # Physics-informed losses and RC circuit simulation
│   └── utils/                # Seeding and CSV/LaTeX export utilities
├── scripts/
│   └── run_pipeline.py       # Main orchestration script
├── artifacts/                # Outputs: trained models, figures, metrics, run_config.yaml
├── archived_notebooks/       # Original exploratory notebooks (do not run)
└── data/                     # Raw, processed, and synthetic data
```

## Setup Environment

This project uses `uv` for dependency management.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Ensure your raw data is placed at `data/raw/all_battery_data_sampled.csv` (configurable in `configs/config.yaml`).

## Running the Pipeline

The end-to-end pipeline is orchestrated by `scripts/run_pipeline.py`. It runs the following stages:
1. Data Preparation (Scaling & Sequence Generation)
2. Autoencoder Pre-training
3. WGAN-GP Training (with Physics Losses)
4. Synthetic Data Generation
5. Evaluation (MMD and KDE/t-SNE/PCA plots)
6. Downstream Task Evaluation (SOH Prediction)
7. Results Export (CSV and LaTeX)

To run the pipeline with the default configuration:

```bash
PYTHONPATH=. python scripts/run_pipeline.py
```

### Configuration Overrides

The pipeline uses `Hydra` for configuration. You can seamlessly override any parameter from `configs/config.yaml` directly from the command line:

```bash
# Run with the advanced physics loss (Energy + Temperature + SOC)
PYTHONPATH=. python scripts/run_pipeline.py physics_mode=advanced

# Run a quick smoke test with reduced epochs
PYTHONPATH=. python scripts/run_pipeline.py training.epochs_gan=10 training.ae_epochs=5 n_synthetic=100

# Run with the RC circuit physics loss
PYTHONPATH=. python scripts/run_pipeline.py physics_mode=rc training.lambda_rc=2.0
```

Every time you run the pipeline, the exact resolved configuration used for that run will be saved to `artifacts/run_config.yaml` to ensure complete reproducibility. All resulting trained models, metric CSVs/LaTeX tables, and figures will be available in the `artifacts/` folder.
